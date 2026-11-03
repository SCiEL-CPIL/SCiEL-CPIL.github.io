"""
Optimization of 100 equal spheres in a unit cube.
Uses a soft-min (log-sum-exp) surrogate for the radius objective with L-BFGS-B,
plus a final high-precision polish pass.
"""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)

    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)

    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r


def _soft_min_radius(centers, beta=200.0):
    """
    Smooth approximation of the minimum radius using log-sum-exp.
    Returns (radius_approx, gradient_flat) for use with L-BFGS-B.
    
    The radius r = min( min_pair_dist/2, min_wall_dist )
    We approximate -log(sum(exp(-beta * constraints))) / beta as a smooth min.
    """
    centers = centers.reshape(-1, 3)
    n = len(centers)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # Upper triangle indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0  # radius from pair
    
    # Wall distances
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    
    # All constraints: pair radii and wall distances
    all_constraints = np.concatenate([pair_dists, wall_dists])
    
    # Soft min: -1/beta * log(sum(exp(-beta * c)))
    # For numerical stability, subtract the min
    min_c = np.min(all_constraints)
    if min_c < 0:
        # If any constraint is violated (negative), penalize heavily
        return min_c, np.zeros(n * 3)
    
    exp_vals = np.exp(-beta * (all_constraints - min_c))
    log_sum = np.log(np.sum(exp_vals))
    soft_min_val = min_c - log_sum / beta
    
    # Gradient computation
    # d(soft_min)/dc_k = -exp(-beta*(c_k - min_c)) / sum(exp(-beta*(c_j - min_c)))
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros(n * 3)
    
    # Pairwise gradient contributions
    # c_pair = |c_i - c_j| / 2
    # d(c_pair)/dc_i = (c_i - c_j) / |c_i - c_j| = (c_i - c_j) / (2 * c_pair)
    # But we have weights on the radius values
    for k in range(len(i_idx)):
        w = weights[k]
        if w < 1e-15:
            continue
        i = i_idx[k]
        j = j_idx[k]
        d = dists[i, j]
        if d < 1e-10:
            continue
        # d(c_pair)/dc_i = (c_i - c_j) / d
        # d(c_pair)/dc_j = (c_j - c_i) / d
        vec = (centers[i] - centers[j]) / d
        grad[i * 3:(i + 1) * 3] += w * vec
        grad[j * 3:(j + 1) * 3] -= w * vec
    
    # Wall gradient contributions
    # c_wall = 0.5 - max(|x|, |y|, |z|)
    # d(c_wall)/dc = -sign(c_k) where c_k is the max abs coordinate
    for i in range(n):
        w = weights[len(pair_dists) + i]
        if w < 1e-15:
            continue
        max_val = max_abs[i]
        for k in range(3):
            if np.isclose(abs(centers[i, k]), max_val, atol=1e-12):
                grad[i * 3 + k] -= w * np.sign(centers[i, k])
                break
    
    return soft_min_val, grad


def _neg_soft_min_radius(x, beta=200.0):
    """
    Negative soft-min radius for minimization.
    x is a flat array of shape (N*3,).
    """
    centers = x.reshape(-1, 3)
    val, grad = _soft_min_radius(centers, beta)
    return -val, -grad


def _init_grid(nx, ny, nz):
    """Create a grid initialization."""
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    
    grid = []
    for z in zs:
        for y in ys:
            for x in xs:
                grid.append([x, y, z])
    return np.array(grid)


def _init_lattice_perturbed(rng, base_spacing):
    """
    Create a perturbed lattice initialization.
    Try to pack as many as possible in a simple cubic or BCC-like arrangement.
    """
    # Simple cubic: spacing s, count per axis ~ 1/s
    # For 100 spheres, try 5x5x4 = 100 with spacing 0.2
    # Or 4x5x5 = 100
    # Let's do a few variants
    
    # Variant 1: 5x5x4
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.45, 0.45, 4)
    centers = []
    for z in zs:
        for y in ys:
            for x in xs:
                centers.append([x + rng.uniform(-0.02, 0.02),
                               y + rng.uniform(-0.02, 0.02),
                               z + rng.uniform(-0.02, 0.02)])
    return np.array(centers)


def _init_random(rng):
    """Random initialization."""
    return rng.uniform(-0.4, 0.4, size=(N, 3))


def _optimize_from(init_centers, beta_schedule=None, time_budget=30.0):
    """
    Optimize from initial centers using soft-min radius surrogate.
    Uses a schedule of decreasing beta for better convergence.
    """
    if beta_schedule is None:
        beta_schedule = [50.0, 100.0, 200.0, 400.0, 800.0]
    
    current_centers = init_centers.copy()
    start_time = time.time()
    
    for beta in beta_schedule:
        if time.time() - start_time > time_budget:
            break
        
        def obj_and_grad(x):
            val, grad = _neg_soft_min_radius(x, beta)
            return val, grad
        
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                obj_and_grad,
                current_centers.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            current_centers = res.x.reshape(-1, 3)
        except Exception:
            pass
        
        # Ensure centers stay within bounds
        current_centers = np.clip(current_centers, -0.499999, 0.499999)
    
    return current_centers


def _polish_centers(centers, time_budget=10.0):
    """
    Final high-precision polish using very high beta.
    """
    start_time = time.time()
    current = centers.copy()
    
    for beta in [1000.0, 2000.0, 5000.0]:
        if time.time() - start_time > time_budget:
            break
        
        def obj_and_grad(x):
            val, grad = _neg_soft_min_radius(x, beta)
            return val, grad
        
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                obj_and_grad,
                current.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'disp': False, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            current = res.x.reshape(-1, 3)
        except Exception:
            pass
        
        current = np.clip(current, -0.499999, 0.499999)
    
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    best_radius = 0.0
    best_centers = None
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    
    # Grid 5x4x5
    inits.append(_init_grid(5, 4, 5))
    # Grid 4x5x5
    inits.append(_init_grid(4, 5, 5))
    # Grid 5x5x4
    inits.append(_init_grid(5, 5, 4))
    
    # Perturbed lattices
    for _ in range(3):
        inits.append(_init_lattice_perturbed(rng, 0.2))
    
    # Random inits
    for _ in range(5):
        inits.append(_init_random(rng))
    
    # Optimize from each initialization
    for init in inits:
        if time.time() > deadline - 20.0:
            break
        
        time_per_init = min(20.0, max(5.0, (deadline - time.time()) / max(1, len(inits) - inits.index(init))))
        
        optimized = _optimize_from(init, time_budget=time_per_init)
        
        r = _compute_radius(optimized)
        if r > best_radius:
            best_radius = r
            best_centers = optimized.copy()
    
    # Polish the best result
    if best_centers is not None:
        if time.time() < deadline - 5.0:
            polished = _polish_centers(best_centers, time_budget=min(10.0, deadline - time.time() - 5.0))
            r_polished = _compute_radius(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished
    
    # Final safety check
    if best_centers is None:
        best_centers = _init_grid(5, 4, 5)
    else:
        best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return best_centers
# EVOLVE-BLOCK-END
