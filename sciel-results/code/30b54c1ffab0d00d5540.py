import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 100
HALF = 0.5
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def _compute_radius(C):
    """
    Compute the maximum common radius for a set of centers C inside [-0.5, 0.5]^3.
    Returns 0.0 if invalid.
    """
    if C.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(C)):
        return 0.0
    
    # Wall constraints
    max_abs = np.max(np.abs(C), axis=1)
    wall_dists = HALF - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < -1e-12:
        return 0.0
    min_wall_dist = max(0.0, min_wall_dist)

    # Pairwise constraints using cKDTree for speed
    tree = cKDTree(C)
    # Query each point against itself and its nearest neighbor
    # k=2 gives self (dist 0) and nearest neighbor
    distances, _ = tree.query(C, k=2)
    min_pair_dist = np.min(distances[:, 1])
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _softplus(x):
    """Numerically stable softplus."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def _energy_and_gradient(r, flat_pos, beta=50.0):
    """
    Compute energy and gradient for fixed radius r.
    Energy is sum of squared softplus penalties for overlaps and boundary violations.
    Uses softplus to ensure smoothness and differentiability everywhere.
    """
    pos = flat_pos.reshape(N, 3)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Boundary distances
    # d_walls[i, k] = HALF - |pos[i, k]|
    # The effective wall distance for sphere i is min_k (HALF - |pos[i, k]|)
    # To make this differentiable, we can use a soft-min or just sum penalties for each axis.
    # Summing penalties for each axis is a valid relaxation (stronger constraint, but easier to optimize).
    # If min_k (HALF - |x_ik|) >= r, then all (HALF - |x_ik|) >= r.
    # So penalizing sum_k softplus(beta * (r - (HALF - |x_ik|)))^2 is fine.
    
    abs_pos = np.abs(pos)
    wall_dists_flat = HALF - abs_pos  # Shape (N, 3)
    
    # Penalties
    # Overlap: 2r - d > 0 means overlap. Penalty: softplus(beta * (2r - d))
    # Boundary: r - d_wall > 0 means violation. Penalty: softplus(beta * (r - d_wall))
    
    overlap_arg = beta * (2.0 * r - pair_dists)
    wall_arg = beta * (r - wall_dists_flat)
    
    sp_overlap = _softplus(overlap_arg)
    sp_wall = _softplus(wall_arg)
    
    energy = np.sum(sp_overlap**2) + np.sum(sp_wall**2)
    
    # Gradients
    # d(softplus(x)^2)/dx = 2 * softplus(x) * sigmoid(x)
    # sigmoid(x) = 1 / (1 + exp(-x))
    
    grad_pos = np.zeros_like(pos)
    
    # Pairwise gradient
    # E_pair = sum sp(overlap_arg)^2
    # dE/dpos_i = sum_j dE/darg_ij * darg_ij/dpos_i
    # dE/darg = 2 * sp * sigmoid
    # darg/dd = -beta
    # dE/dd = -2 * beta * sp * sigmoid
    # d(d_ij)/dpos_i = (pos_i - pos_j) / d_ij
    
    sig_overlap = 1.0 / (1.0 + np.exp(-overlap_arg))
    # coeff_ij = dE/darg * darg/dd * (1/d_ij) = 2 * sp * sig * (-beta) / d_ij
    # Note: dE/dpos_i = sum_j coeff_ij * (pos_i - pos_j)
    # dE/dpos_j = -sum_i coeff_ij * (pos_i - pos_j)
    
    # Vectorized calculation for pairwise forces
    # We need to accumulate forces on each particle.
    # F_i = sum_{j != i} (2 * beta * sp_ij * sig_ij / d_ij) * (pos_i - pos_j)
    
    # Create a matrix of coefficients W where W[i,j] = 2 * beta * sp_ij * sig_ij / d_ij
    # We only have upper triangle values.
    W = np.zeros((N, N))
    
    # Avoid division by zero
    valid_mask = pair_dists > 1e-12
    if np.any(valid_mask):
        iu_valid = (iu[0][valid_mask], iu[1][valid_mask])
        d_valid = pair_dists[valid_mask]
        sp_valid = sp_overlap[valid_mask]
        sig_valid = sig_overlap[valid_mask]
        
        coeff_valid = 2.0 * beta * sp_valid * sig_valid / d_valid
        W[iu_valid[0], iu_valid[1]] = coeff_valid
        W[iu_valid[1], iu_valid[0]] = coeff_valid
    
    # Force on i: sum_j W[i,j] * (pos_i - pos_j)
    # = pos_i * sum_j W[i,j] - sum_j W[i,j] * pos_j
    sum_W = np.sum(W, axis=1)
    term1 = pos * sum_W[:, np.newaxis]
    term2 = W @ pos
    grad_pos += term1 - term2
    
    # Boundary gradient
    # E_wall = sum_{i,k} sp(wall_arg_ik)^2
    # dE/dpos_ik = dE/darg * darg/dpos_ik
    # dE/darg = 2 * sp * sigmoid
    # darg/dpos_ik = beta * d(r - (HALF - |x_ik|))/dpos_ik = beta * sign(x_ik)
    # So dE/dpos_ik = 2 * beta * sp * sig * sign(x_ik)
    
    sig_wall = 1.0 / (1.0 + np.exp(-wall_arg))
    signs = np.sign(pos)
    # Handle zero sign
    signs[signs == 0] = 1.0
    
    grad_wall_contrib = 2.0 * beta * sp_wall * sig_wall * signs
    grad_pos += grad_wall_contrib
    
    return energy, grad_pos.flatten()

def _minimize_energy(r, pos, max_iter=200, ftol=1e-12, gtol=1e-10):
    """Minimize energy for fixed radius r."""
    bounds = [(-HALF, HALF)] * (N * 3)
    try:
        res = minimize(
            lambda x: _energy_and_gradient(r, x)[0],
            pos.flatten(),
            jac=lambda x: _energy_and_gradient(r, x)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if np.isfinite(res.fun):
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def _generate_initial_points(seed):
    """Generate diverse initial points using FCC-like and BCC-like structures."""
    rng = np.random.default_rng(seed)
    method = seed % 3
    
    if method == 0:
        # FCC-like: 4 points per unit cell
        size = 4
        raw = []
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    raw.append([i, j, k])
                    raw.append([i + 0.5, j + 0.5, k])
                    raw.append([i + 0.5, j, k + 0.5])
                    raw.append([i, j + 0.5, k + 0.5])
        raw = np.array(raw)
        center = np.array([size/2 - 0.25, size/2 - 0.25, size/2 - 0.25])
        raw = raw - center
        max_coord = np.max(np.abs(raw))
        scale = 0.45 / max_coord
        raw *= scale
        idx = rng.permutation(len(raw))
        return raw[idx[:N]].copy()
    elif method == 1:
        # BCC-like: 2 points per unit cell
        size = 4
        raw = []
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    raw.append([i, j, k])
                    raw.append([i + 0.5, j + 0.5, k + 0.5])
        raw = np.array(raw)
        center = np.array([size/2 - 0.25, size/2 - 0.25, size/2 - 0.25])
        raw = raw - center
        max_coord = np.max(np.abs(raw))
        scale = 0.45 / max_coord
        raw *= scale
        idx = rng.permutation(len(raw))
        return raw[idx[:N]].copy()
    else:
        # Random
        return rng.uniform(-0.45, 0.45, (N, 3))

def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - SAFETY_MARGIN)
    
    best_radius = 0.0
    best_centers = None
    
    # Start with a reasonable radius estimate
    # BCC packing density in cube is roughly 0.68.
    # Volume of 100 spheres = 100 * 4/3 pi r^3.
    # 100 * 4/3 pi r^3 <= 0.68 * 1
    # r^3 <= 0.68 / (400/3 pi) = 0.68 / 418.8 = 0.00162
    # r <= 0.117
    # Let's start at 0.10 to be safe.
    r_current = 0.10
    
    # Try multiple seeds
    seeds = [42, 123, 456, 789, 101112, 202021, 303031, 404041, 505051, 606061]
    
    for seed in seeds:
        if _check_time():
            break
            
        centers = _generate_initial_points(seed)
        centers = np.clip(centers, -HALF + 1e-9, HALF - 1e-9)
        
        # Iterative Radius Increase
        r_try = r_current
        improved = False
        max_iter_r = 30 # Number of radius increase steps
        
        for _ in range(max_iter_r):
            if _check_time():
                break
                
            # Minimize energy for current r_try
            # Use more iterations as we get closer to the limit
            max_iter_opt = 100
            if r_try > 0.11:
                max_iter_opt = 200
            if r_try > 0.115:
                max_iter_opt = 300
                
            optimized_centers = _minimize_energy(r_try, centers, max_iter=max_iter_opt)
            
            # Check if feasible
            actual_radius = _compute_radius(optimized_centers)
            
            if actual_radius >= r_try - 1e-6:
                # Feasible! Increase radius
                r_try += 0.001
                centers = optimized_centers
                improved = True
            else:
                # Not feasible, stop increasing
                break
        
        # Final polish with the highest feasible radius found for this seed
        # Use the actual radius achieved
        final_radius = _compute_radius(centers)
        
        if final_radius > best_radius:
            best_radius = final_radius
            best_centers = centers.copy()
            
        # Update global r_current if we did well
        if best_radius > r_current:
            r_current = best_radius

    # If we have time, do a final polish on the best solution
    if best_centers is not None and not _check_time():
        # Try to push a tiny bit further
        push_r = best_radius + 0.0005
        polished = _minimize_energy(push_r, best_centers, max_iter=500, ftol=1e-14, gtol=1e-12)
        polished_r = _compute_radius(polished)
        if polished_r > best_radius:
            best_radius = polished_r
            best_centers = polished
            
        # Final micro-polish
        if not _check_time():
            polished2 = _minimize_energy(best_radius, best_centers, max_iter=500, ftol=1e-15, gtol=1e-14)
            polished2_r = _compute_radius(polished2)
            if polished2_r > best_radius:
                best_radius = polished2_r
                best_centers = polished2

    if best_centers is None:
        # Fallback: return a grid
        grid = np.linspace(-0.4, 0.4, 10)
        centers = np.array([(x, y, z) for z in grid for y in grid for x in grid])
        return centers[:N]
        
    return best_centers
