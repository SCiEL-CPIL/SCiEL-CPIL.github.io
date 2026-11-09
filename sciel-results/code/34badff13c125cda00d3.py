"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
CUBE_HALF = 0.5

# EVOLVE-BLOCK-START

def _pairwise_distances(pts):
    """Compute all pairwise distances for N points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))

def _score(pts):
    """Compute the packing radius for given centers."""
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check bounds
    if np.any(pts < -CUBE_HALF - 1e-12) or np.any(pts > CUBE_HALF + 1e-12):
        # Clamp for scoring but flag invalid
        pass
    # Distance to walls
    wall_dists = np.min(np.min(CUBE_HALF - np.abs(pts), axis=1), axis=0)
    # Pairwise distances
    pdist = _pairwise_distances(pts)
    np.fill_diagonal(pdist, np.inf)
    min_pair = np.min(pdist)
    return min(wall_dists, min_pair / 2.0)

def _objective_and_grad(pts):
    """
    Smooth surrogate objective: minimize -soft_min of all constraints.
    We use a log-sum-exp based soft-min to get gradients.
    
    The actual radius r = min(wall_dists, pair_dists/2).
    We want to maximize r, so minimize -r.
    Using soft-min: soft_min(x) = -log(sum(exp(-beta * x))) / beta
    For large beta, this approaches min(x).
    """
    beta = 50.0  # sharpness of soft-min
    
    # Wall distances: for each point i and each coordinate k, dist = 0.5 - |c_ik|
    # We need the min over all i,k
    abs_pts = np.abs(pts)
    wall_dists_flat = CUBE_HALF - abs_pts  # shape (N, 3)
    
    # Pairwise distances / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    pair_dists_flat = dists[~np.eye(N, dtype=bool)]
    
    # Combine all constraints into one vector
    all_dists = np.concatenate([wall_dists_flat.ravel(), pair_dists_flat])
    
    # Soft-min
    # To avoid overflow, subtract max
    max_d = np.max(all_dists)
    exp_neg = np.exp(-beta * (all_dists - max_d))
    log_sum = np.log(np.sum(exp_neg))
    soft_min = max_d - log_sum / beta
    
    # Objective: minimize -soft_min (i.e., maximize soft_min)
    obj = -soft_min
    
    # Gradient
    # d(soft_min)/dx = sum_i exp(-beta*(x_i - max_d)) * (-beta) / sum_j exp(-beta*(x_j - max_d))
    #                = -beta * exp_neg / sum(exp_neg)
    # So d(obj)/dx = d(-soft_min)/dx = beta * exp_neg / sum(exp_neg)
    # But we need to map back to pts coordinates.
    
    grad = np.zeros_like(pts)
    weights = exp_neg / np.sum(exp_neg)
    
    # Wall distances: for point i, coordinate k, the distance is 0.5 - |c_ik|
    # d(dist)/d(c_ik) = -sign(c_ik)
    # So d(obj)/d(c_ik) = weights_wall[i,k] * beta * (-sign(c_ik))
    wall_weights = weights[:N*3].reshape(N, 3)
    signs = np.sign(pts)
    # Handle zero: gradient is undefined, use 0
    signs[pts == 0] = 0
    grad -= beta * wall_weights * signs  # because obj = -soft_min, grad of obj = -grad of soft_min
    # Wait: soft_min = max_d - log_sum/beta
    # d(soft_min)/dx_i = -beta * exp(-beta*(x_i - max_d)) / sum(exp_neg) = -beta * weights[i]
    # So d(obj)/dx_i = -d(soft_min)/dx_i = beta * weights[i]
    # And d(x_i)/d(pts) for wall: d(0.5 - |c_ik|)/d(c_ik) = -sign(c_ik)
    # So d(obj)/d(c_ik) = beta * weights[i,k] * (-sign(c_ik))
    # That's what I have: grad -= beta * wall_weights * signs
    # But wait, the sign of the gradient for minimization:
    # We want to increase soft_min, so we should move in direction of +grad(soft_min)
    # grad(soft_min) = -beta * weights (for each constraint)
    # For wall constraint: d(soft_min)/d(c_ik) = -beta * weights[i,k] * (-sign(c_ik)) = beta * weights[i,k] * sign(c_ik)
    # So to maximize soft_min, move in direction beta * weights[i,k] * sign(c_ik)
    # For minimization of obj = -soft_min, gradient is -grad(soft_min) = -beta * weights[i,k] * sign(c_ik)
    # So grad[pts] for walls should be -beta * wall_weights * signs
    # That matches what I wrote.
    
    # Pairwise distances: for pair (i,j), dist = |c_i - c_j| / 2
    # d(dist)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    # d(dist)/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|)
    pair_weights = weights[N*3:]
    # Reconstruct pairs
    pair_idx = np.argwhere(~np.eye(N, dtype=bool))
    for idx, (i, j) in enumerate(pair_idx):
        w = pair_weights[idx]
        if w > 0:
            diff_ij = pts[i] - pts[j]
            dist_ij = np.linalg.norm(diff_ij)
            if dist_ij > 1e-12:
                # d(dist)/d(c_i) = diff_ij / (2 * dist_ij)
                # d(obj)/d(c_i) = beta * w * diff_ij / (2 * dist_ij)
                # Wait, let me redo.
                # soft_min includes x_ij = |c_i - c_j|/2
                # d(soft_min)/dx_ij = -beta * weights[idx]
                # d(x_ij)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
                # d(x_ij)/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|)
                # So d(soft_min)/d(c_i) = -beta * weights[idx] * (c_i - c_j) / (2 * |c_i - c_j|)
                # d(obj)/d(c_i) = -d(soft_min)/d(c_i) = beta * weights[idx] * (c_i - c_j) / (2 * |c_i - c_j|)
                grad[i] += beta * w * diff_ij / (2 * dist_ij)
                grad[j] -= beta * w * diff_ij / (2 * dist_ij)
    
    return obj, grad

def _project_to_cube(pts):
    """Project points into the cube [-0.5, 0.5]^3."""
    return np.clip(pts, -CUBE_HALF, CUBE_HALF)

def _initial_grid():
    """Create a simple cubic grid initial guess."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].astype(float)

def construct():
    """
    Construct 50 sphere centers in the unit cube to maximize the minimum radius.
    Uses gradient-based optimization with a smooth surrogate objective.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    # Initial guess: simple grid
    best_pts = _initial_grid()
    best_score = _score(best_pts)
    
    # Try multiple random restarts and refinement passes
    rng = np.random.RandomState(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Generate initial guess
        if restart == 0:
            pts = _initial_grid()
        else:
            # Random points in the cube, with some jitter around grid
            if restart % 2 == 0:
                # Perturbed grid
                pts = _initial_grid() + 0.05 * rng.randn(N, 3)
            else:
                # Random
                pts = rng.uniform(-0.4, 0.4, (N, 3))
            pts = _project_to_cube(pts)
        
        # Optimize using L-BFGS-B with bounds
        bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
        
        max_iters = 500
        try:
            result = minimize(
                _objective_and_grad,
                pts,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            pts_opt = _project_to_cube(result.x)
        except Exception:
            pts_opt = pts
        
        score_opt = _score(pts_opt)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
        
        # If we have time, do a polish pass with higher precision
        if time.time() < deadline - 20:
            # Polish with more iterations and tighter tolerance
            try:
                result2 = minimize(
                    _objective_and_grad,
                    best_pts,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                pts_polish = _project_to_cube(result2.x)
                score_polish = _score(pts_polish)
                if score_polish > best_score:
                    best_score = score_polish
                    best_pts = pts_polish
            except Exception:
                pass
        
        # Early stop if we're doing well
        if best_score > 0.135:
            # Do one more polish
            if time.time() < deadline - 10:
                try:
                    result3 = minimize(
                        _objective_and_grad,
                        best_pts,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    pts_final = _project_to_cube(result3.x)
                    score_final = _score(pts_final)
                    if score_final > best_score:
                        best_score = score_final
                        best_pts = pts_final
                except Exception:
                    pass
            break
    
    return best_pts

# EVOLVE-BLOCK-END
