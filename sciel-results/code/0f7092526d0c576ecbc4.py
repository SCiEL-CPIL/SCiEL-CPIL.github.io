"""Circle packing in unit square: pack 26 circles with disjoint interiors, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

def _compute_objective(params, n=N):
    """
    params: array of shape (3*n,) = [x1, y1, r1, x2, y2, r2, ...]
    Returns: (sum_radii, valid)
    """
    pts = params.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Check boundaries
    if np.any(rs <= 0):
        return 0.0, False
    if np.any(xs < rs) or np.any(xs + rs > 1.0):
        return 0.0, False
    if np.any(ys < rs) or np.any(ys + rs > 1.0):
        return 0.0, False
    
    # Check pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq < (rs[i] + rs[j]) ** 2 - 1e-9:
                return 0.0, False
    
    return np.sum(rs), True

def _penalized_objective(params, n=N, mu=10.0):
    """
    Smooth penalized objective for optimization.
    We maximize sum(r) - mu * penalty(violations)
    So we minimize -sum(r) + mu * penalty
    """
    pts = params.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Ensure positive radii (soft)
    r_pos = np.maximum(rs, 1e-6)
    
    obj = -np.sum(r_pos)
    
    # Boundary violations
    # x < r => violation = max(0, r - x)
    # x + r > 1 => violation = max(0, x + r - 1)
    bound_viol = np.maximum(0, r_pos - xs) + np.maximum(0, xs + r_pos - 1.0)
    bound_viol += np.maximum(0, r_pos - ys) + np.maximum(0, ys + r_pos - 1.0)
    obj += mu * np.sum(bound_viol)
    
    # Pairwise overlap violations
    # dist < r_i + r_j => violation = max(0, r_i + r_j - dist)
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            viol = np.maximum(0, r_pos[i] + r_pos[j] - dist)
            obj += mu * viol
    
    return obj

def _make_initial_grid(n=N):
    """Create a grid-based initial guess."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append((x, y, r))
    return np.array(cells)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_params = _make_initial_grid(N)
    best_sum, _ = _compute_objective(best_params)
    
    if best_sum <= 0:
        # Fallback: tiny circles
        best_params = np.zeros((N, 3))
        for i in range(N):
            best_params[i] = [0.5 + 0.01 * np.cos(2 * np.pi * i / N),
                              0.5 + 0.01 * np.sin(2 * np.pi * i / N),
                              0.01]
        best_sum, _ = _compute_objective(best_params)
    
    # Multiple restarts with perturbation
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize with grid + random perturbation
        if restart == 0:
            params = best_params.flatten().copy()
        else:
            params = _make_initial_grid(N).flatten().copy()
            # Add random perturbation
            noise_scale = 0.05 / (1 + restart * 0.1)
            params += np.random.randn(len(params)) * noise_scale
            # Ensure radii are positive
            for k in range(N):
                params[3 * k + 2] = max(params[3 * k + 2], 0.01)
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _penalized_objective,
                params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < _penalized_objective(params):
                new_params = result.x
                new_sum, valid = _compute_objective(new_params)
                if valid and new_sum > best_sum:
                    best_sum = new_sum
                    best_params = new_params.reshape(N, 3)
        except Exception:
            continue
        
        if time.time() > deadline - 15.0:
            break
    
    # Final polish: local refinement with tighter tolerances
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                _penalized_objective,
                best_params.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 20000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < _penalized_objective(best_params.flatten()):
                new_params = result.x
                new_sum, valid = _compute_objective(new_params)
                if valid and new_sum > best_sum:
                    best_sum = new_sum
                    best_params = new_params.reshape(N, 3)
        except Exception:
            pass
    
    # Ensure final result is valid
    final_sum, valid = _compute_objective(best_params)
    if not valid:
        # Fallback to a simple valid configuration
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 1e-3
                    cells.append((x, y, r))
        best_params = np.array(cells[:N])
    
    return best_params
