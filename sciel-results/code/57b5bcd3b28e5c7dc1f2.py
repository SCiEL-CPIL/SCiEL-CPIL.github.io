"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# Best known packing radius (Pfoertner) for 50 spheres in unit cube
R_BEST = 0.1359545126417


def _score(centers):
    """Compute the achievable radius for a set of centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside cube [-0.5, 0.5]^3
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    # Compute minimum pairwise distance
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist <= 0:
        return 0.0
    # Compute minimum distance to boundary
    boundary_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(boundary_dists)
    r = min(min_dist / 2.0, min_boundary)
    return max(r, 0.0)


def _soft_min(x, beta=50.0):
    """Soft minimum using log-sum-exp trick."""
    x = np.asarray(x)
    m = np.min(x)
    return m - np.log(np.sum(np.exp(-beta * (x - m)))) / beta


def _objective(centers, target_r, beta=30.0):
    """
    Objective: minimize violation of constraints.
    We want all pairwise distances >= 2*target_r and all centers within [target_r, 0.5-target_r].
    """
    centers = np.asarray(centers).reshape(-1, 3)
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    # Violation: 2*target_r - dist (positive means violation)
    pair_viol = 2.0 * target_r - dists
    # Boundary violation: center - target_r (positive means outside inner box)
    # For each coordinate, violation is max(0, |c| - (0.5 - target_r))
    boundary_viol = np.abs(centers) - (0.5 - target_r)
    boundary_viol = np.maximum(boundary_viol, 0.0)
    
    # Use soft-min to get a smooth measure of the worst violation
    # We want to minimize the maximum violation
    all_viol = np.concatenate([pair_viol.ravel(), boundary_viol.ravel()])
    # Soft max of violations
    m = np.max(all_viol)
    soft_max = m + np.log(np.sum(np.exp(beta * (all_viol - m)))) / beta
    return soft_max


def _make_initial_grid(n_spheres):
    """Create a reasonable initial grid packing."""
    m = int(np.ceil(n_spheres ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n_spheres].copy()


def _local_optimize(centers, target_r, max_iter=200, tol=1e-10):
    """
    Local optimization: push centers apart to satisfy constraints.
    Uses gradient descent on the soft-max violation objective.
    """
    centers = centers.copy()
    n = centers.shape[0]
    
    best_centers = centers.copy()
    best_viol = _objective(centers, target_r)
    
    for it in range(max_iter):
        if best_viol < tol:
            break
        
        # Compute gradient numerically (or analytically)
        # Analytical gradient of soft-max violation
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        pair_viol = 2.0 * target_r - dists
        
        boundary_viol = np.abs(centers) - (0.5 - target_r)
        boundary_viol = np.maximum(boundary_viol, 0.0)
        
        all_viol = np.concatenate([pair_viol.ravel(), boundary_viol.ravel()])
        beta = 30.0
        m = np.max(all_viol)
        exps = np.exp(beta * (all_viol - m))
        weights = exps / np.sum(exps)
        
        # Gradient w.r.t. centers
        grad = np.zeros_like(centers)
        
        # Pairwise contribution
        pair_weights = weights[:n*n].reshape(n, n)
        # grad_i = -sum_j w_ij * (c_i - c_j) / |c_i - c_j| * beta
        # Actually d/dc_i (2r - |c_i - c_j|) = -(c_i - c_j)/|c_i - c_j|
        # So contribution to grad from pair (i,j) is -w_ij * (c_i - c_j)/|c_i - c_j|
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist > 1e-12:
                    grad[i] += -pair_weights[i, j] * d / dist
                grad[j] += -pair_weights[j, i] * (-d) / dist
        
        # Boundary contribution
        bnd_weights = weights[n*n:]
        # d/dc_i max(0, |c_ik| - (0.5-r)) 
        # If |c_ik| > (0.5-r), derivative is sign(c_ik)
        for i in range(n):
            for k in range(3):
                if np.abs(centers[i, k]) > (0.5 - target_r):
                    grad[i, k] += bnd_weights[i*3+k] * np.sign(centers[i, k])
        
        # Gradient descent
        lr = 0.01
        centers_new = centers - lr * grad
        # Clip to cube
        centers_new = np.clip(centers_new, -0.5, 0.5)
        
        viol_new = _objective(centers_new, target_r)
        if viol_new < best_viol:
            best_viol = viol_new
            best_centers = centers_new.copy()
            centers = centers_new
        else:
            # Reduce learning rate
            lr *= 0.5
            if lr < 1e-6:
                break
    
    return best_centers, best_viol


def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Start with a grid
    centers = _make_initial_grid(N)
    current_r = _score(centers)
    
    # Binary search for maximum feasible radius
    # We know r is at most 0.5 (single sphere) and at least current_r
    r_low = current_r
    r_high = 0.25  # Upper bound estimate
    
    # First, try to increase radius from the grid
    best_centers = centers.copy()
    best_r = current_r
    
    # Try several target radii in increasing order
    # Use a sequence of targets between current_r and R_BEST
    if current_r < R_BEST:
        targets = np.linspace(current_r, R_BEST, num=8)
    else:
        targets = np.array([current_r * 1.01])
    
    for target_r in targets:
        if time.time() > deadline - 30:
            break
        
        if target_r <= best_r:
            continue
        
        # Try to find a configuration that achieves target_r
        # Start from best_centers
        opt_centers, viol = _local_optimize(best_centers, target_r, max_iter=100)
        
        # Check if we achieved it
        if viol < 1e-6:
            actual_r = _score(opt_centers)
            if actual_r >= target_r - 1e-9:
                best_centers = opt_centers.copy()
                best_r = actual_r
        
        # Also try from a fresh grid perturbed
        if time.time() < deadline - 60:
            rng = np.random.RandomState(int(target_r * 1e6))
            perturbed = best_centers + rng.normal(0, 0.01, size=(N, 3))
            perturbed = np.clip(perturbed, -0.5, 0.5)
            opt_centers2, viol2 = _local_optimize(perturbed, target_r, max_iter=100)
            if viol2 < 1e-6:
                actual_r2 = _score(opt_centers2)
                if actual_r2 > best_r:
                    best_centers = opt_centers2.copy()
                    best_r = actual_r
    
    # Final polish: try to squeeze out more radius
    if time.time() < deadline - 10 and best_r < R_BEST:
        # Try a slightly higher target
        target = min(best_r * 1.005, R_BEST)
        opt_centers, viol = _local_optimize(best_centers, target, max_iter=150)
        if viol < 1e-6:
            actual_r = _score(opt_centers)
            if actual_r > best_r:
                best_centers = opt_centers.copy()
                best_r = actual_r
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    final_r = _score(best_centers)
    
    if final_r <= 0 or not np.all(np.isfinite(best_centers)):
        # Fallback to grid
        return _make_initial_grid(N)
    
    return best_centers
# EVOLVE-BLOCK-END
