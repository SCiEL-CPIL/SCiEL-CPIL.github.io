"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
import random

N = 26
TIME_LIMIT = 150.0  # seconds (evaluator kills at 150s)
SAFETY_MARGIN = 5.0  # seconds to return before limit


def _make_grid_initial():
    """Create a 6x6 grid initial configuration using 26 cells."""
    side = 6
    positions = []
    radii = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side
                positions.append([x, y])
                radii.append(r)
    return np.array(positions), np.array(radii)


def _make_hex_initial():
    """Create a hexagonal packing initial guess."""
    pos = []
    rad = []
    rows = [5, 5, 5, 5, 6]  # total = 26
    for k, count in enumerate(rows):
        y = (k + 0.5) / len(rows)
        if k % 2 == 0:
            xs = [(i + 0.5) / count for i in range(count)]
        else:
            xs = [(i + 0.5) / count for i in range(count)]
        for x in xs:
            if len(pos) < N:
                pos.append([x, y])
                rad.append(0.08)
    return np.array(pos), np.array(rad)


def _make_random_initial(seed=42):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    rad = np.full(N, 0.05)
    return pos, rad


def _vectorized_objective(params, beta=20.0):
    """
    Vectorized objective with smooth penalty for constraint violations.
    Uses soft-min formulation for non-overlap constraints.
    """
    pos = params[:2*N].reshape(N, 2)
    rad = params[2*N:]
    
    # Boundary violations
    left_viol = np.maximum(0, -pos[:, 0] + rad)
    right_viol = np.maximum(0, pos[:, 0] + rad - 1)
    bottom_viol = np.maximum(0, -pos[:, 1] + rad)
    top_viol = np.maximum(0, pos[:, 1] + rad - 1)
    
    boundary_penalty = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                       np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # Non-overlap violations (vectorized)
    # For each pair (i,j), compute distance and required distance
    # Use broadcasting for all pairs
    dx = pos[:, 0:1] - pos[1:, 0:1].T  # (N-1, N)
    dy = pos[:, 1:2] - pos[1:, 1:2].T
    dist = np.sqrt(dx**2 + dy**2)
    
    # Required distance = rad[i] + rad[j]
    req_dist = rad[:, None] + rad[None, :]
    
    # Only consider upper triangle (i < j)
    mask = np.triu(np.ones((N-1, N)), k=0).astype(bool)
    # Actually we need pairs (i, j) where i < j, so indices 0..N-2 for first, 1..N-1 for second
    # Let's redo: pos[i] and pos[j] for i < j
    # Use full N x N matrix and take upper triangle
    dx_full = pos[:, 0:1] - pos[:, 0:1].T  # (N, N)
    dy_full = pos[:, 1:2] - pos[:, 1:2].T
    dist_full = np.sqrt(dx_full**2 + dy_full**2)
    req_full = rad[:, None] + rad[None, :]
    
    # Upper triangle mask
    tri_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0, req_full - dist_full)
    overlap_penalty = np.sum(viol[tri_mask]**2)
    
    penalty = 100.0 * (boundary_penalty + overlap_penalty)
    
    return -np.sum(rad) + penalty


def _vectorized_gradient(params, beta=20.0):
    """
    Vectorized gradient of the objective function.
    """
    pos = params[:2*N].reshape(N, 2)
    rad = params[2*N:]
    
    grad = np.zeros_like(params)
    
    # Boundary gradient contributions
    # d/dx_i of (x_i - r_i)^2 when x_i - r_i < 0: 2*(x_i - r_i)
    # d/dx_i of (x_i + r_i - 1)^2 when x_i + r_i > 1: 2*(x_i + r_i - 1)
    # etc.
    
    left_viol = np.maximum(0, -pos[:, 0] + rad)
    right_viol = np.maximum(0, pos[:, 0] + rad - 1)
    bottom_viol = np.maximum(0, -pos[:, 1] + rad)
    top_viol = np.maximum(0, pos[:, 1] + rad - 1)
    
    # Gradient w.r.t. x
    grad[:N] += 200.0 * np.where(left_viol > 0, -left_viol, 0)  # d/dx of (-x+r)^2 = -2(-x+r)
    grad[:N] += 200.0 * np.where(right_viol > 0, right_viol, 0)   # d/dx of (x+r-1)^2 = 2(x+r-1)
    
    # Gradient w.r.t. y
    grad[N:2*N] += 200.0 * np.where(bottom_viol > 0, -bottom_viol, 0)
    grad[N:2*N] += 200.0 * np.where(top_viol > 0, top_viol, 0)
    
    # Gradient w.r.t. r from boundary
    # d/dr of (-x+r)^2 = 2(-x+r), d/dr of (x+r-1)^2 = 2(x+r-1), etc.
    grad[2*N:] += 200.0 * np.where(left_viol > 0, -left_viol, 0)
    grad[2*N:] += 200.0 * np.where(right_viol > 0, right_viol, 0)
    grad[2*N:] += 200.0 * np.where(bottom_viol > 0, -bottom_viol, 0)
    grad[2*N:] += 200.0 * np.where(top_viol > 0, top_viol, 0)
    
    # Non-overlap gradient
    dx_full = pos[:, 0:1] - pos[:, 0:1].T
    dy_full = pos[:, 1:2] - pos[:, 1:2].T
    dist_full = np.sqrt(dx_full**2 + dy_full**2)
    req_full = rad[:, None] + rad[None, :]
    
    tri_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    viol = np.maximum(0, req_full - dist_full)
    
    # d/dx_i of (req - dist)^2 = -2*(req-dist) * d(dist)/d(x_i)
    # d(dist)/d(x_i) = (x_i - x_j)/dist for pair (i,j)
    
    # For each pair (i,j) with i<j and viol > 0:
    # grad_x[i] += -2*viol * (x_i - x_j)/dist
    # grad_x[j] += -2*viol * (x_j - x_i)/dist = 2*viol * (x_i - x_j)/dist
    # grad_r[i] += 2*viol, grad_r[j] += 2*viol
    
    # Vectorized approach:
    # For x component:
    # Contribution to grad_x[i] from all j: sum over j of -2*viol[i,j] * (x_i - x_j)/dist[i,j]
    # We can compute this as: for each i, sum_j where j>i
    
    # Let's compute the full gradient contribution matrix
    # G_x[i,j] = -2*viol[i,j] * (x_i - x_j)/dist[i,j] for i!=j, else 0
    # Then grad_x[i] = sum_j G_x[i,j]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist = np.where(dist_full > 1e-12, 1.0/dist_full, 0.0)
    
    # G_x[i,j] = -2*viol[i,j] * (x_i - x_j) * inv_dist[i,j]
    G_x = -2.0 * viol * dx_full * inv_dist
    G_y = -2.0 * viol * dy_full * inv_dist
    
    # Only use upper triangle
    G_x = G_x * tri_mask
    G_y = G_y * tri_mask
    
    grad[:N] += 200.0 * np.sum(G_x, axis=1)
    grad[N:2*N] += 200.0 * np.sum(G_y, axis=1)
    
    # Gradient w.r.t. r: each violation contributes 2*viol to both r_i and r_j
    # grad_r[i] = sum_j 2*viol[i,j] for j where pair (i,j) exists
    grad[2*N:] += 200.0 * np.sum(2.0 * viol * tri_mask, axis=1)
    
    # Gradient w.r.t. r from objective: -1
    grad[2*N:] -= 1.0
    
    return grad


def _enforce_validity(pos, rad, iterations=50):
    """
    Enforce validity by adjusting radii to ensure no overlaps and boundary compliance.
    """
    rad = rad.copy()
    
    for _ in range(iterations):
        changed = False
        
        # Boundary clamp
        for i in range(N):
            max_r = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
            if rad[i] > max_r:
                rad[i] = max(0, max_r)
                changed = True
        
        # Non-overlap enforcement
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-12:
                    continue
                if rad[i] + rad[j] > dist - 1e-9:
                    scale = max(0, (dist - 1e-9)) / max(rad[i] + rad[j], 1e-12)
                    rad[i] *= scale
                    rad[j] *= scale
                    changed = True
        
        if not changed:
            break
    
    # Final boundary clamp
    for i in range(N):
        max_r = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
        rad[i] = min(rad[i], max_r)
        rad[i] = max(rad[i], 0)
    
    return rad


def _is_valid(pos, rad, tol=1e-9):
    """Check if the configuration is valid."""
    for i in range(N):
        x, y, r = pos[i, 0], pos[i, 1], rad[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1 + tol:
            return False
        if y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < rad[i] + rad[j] - tol:
                return False
    return True


def _score(pos, rad):
    """Compute the score (sum of radii) if valid, else 0."""
    if _is_valid(pos, rad):
        return np.sum(rad)
    return 0.0


def construct():
    """Construct 26 disjoint circles in [0,1]^2 maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # Try multiple initial configurations
    configs = [
        _make_grid_initial(),
        _make_hex_initial(),
        _make_random_initial(42),
        _make_random_initial(123),
        _make_random_initial(999),
    ]
    
    best_params = None
    best_score = float('-inf')
    best_pos = None
    best_rad = None
    
    for cfg_pos, cfg_rad in configs:
        if time.time() > deadline:
            break
        
        x0 = np.concatenate([cfg_pos.flatten(), cfg_rad])
        
        # Try L-BFGS-B with gradient
        if time.time() < deadline:
            try:
                res = minimize(
                    _vectorized_objective,
                    x0,
                    jac=_vectorized_gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x
            except Exception:
                pass
        
        # Try Nelder-Mead for local refinement (no gradient)
        if time.time() < deadline:
            try:
                start_params = best_params if best_params is not None else x0
                res_nm = minimize(
                    _vectorized_objective,
                    start_params,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if res_nm.fun < best_score:
                    best_score = res_nm.fun
                    best_params = res_nm.x
            except Exception:
                pass
        
        # Extract and validate
        if best_params is not None:
            pos = best_params[:2*N].reshape(N, 2)
            rad = best_params[2*N:]
            rad = _enforce_validity(pos, rad)
            
            if _is_valid(pos, rad):
                current_score = np.sum(rad)
                if current_score > best_score:
                    best_score = current_score
                    best_pos = pos.copy()
                    best_rad = rad.copy()
    
    # If no valid solution found, use fallback grid
    if best_pos is None:
        pos, rad = _make_grid_initial()
        rad = _enforce_validity(pos, rad)
        best_pos = pos
        best_rad = rad
    
    result = np.column_stack([best_pos[:, 0], best_pos[:, 1], best_rad])
    return result
