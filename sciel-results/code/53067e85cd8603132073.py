"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _time_remaining(deadline):
    return deadline - time.time()


def _validate(x, y, r, w, h, tol=1e-5):
    if w + h > 2.0 + 1e-4:
        return False
    if w < 1e-8 or h < 1e-8:
        return False
    for i in range(len(r)):
        if r[i] < 1e-8:
            return False
        if x[i] - r[i] < -tol or x[i] + r[i] > w + tol:
            return False
        if y[i] - r[i] < -tol or y[i] + r[i] > h + tol:
            return False
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            rad_sum = r[i] + r[j]
            if dist_sq < rad_sum * rad_sum - tol:
                return False
    return True


def _objective(v, w, h, lam=1e4):
    """Penalty-based objective: maximize sum of radii subject to constraints."""
    n = len(v) // 3 - 2  # v has 2*N + 2 elements (x, y, r, w, h) but we fix w,h
    # Actually v will be [x0...xN-1, y0...yN-1, r0...rN-1] with w,h fixed externally
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:3*N]
    
    # Boundary penalties
    penalty = 0.0
    for i in range(N):
        # x bounds
        if x[i] - r[i] < 0:
            penalty += lam * (x[i] - r[i]) ** 2
        if x[i] + r[i] > w:
            penalty += lam * (x[i] + r[i] - w) ** 2
        # y bounds
        if y[i] - r[i] < 0:
            penalty += lam * (y[i] - r[i]) ** 2
        if y[i] + r[i] > h:
            penalty += lam * (y[i] + r[i] - h) ** 2
        # radius positive
        if r[i] < 0:
            penalty += lam * r[i] ** 2
    
    # Non-overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            rad_sum = r[i] + r[j]
            if dist_sq < rad_sum * rad_sum:
                gap = rad_sum - np.sqrt(dist_sq)
                penalty += lam * gap ** 2
    
    # Objective: maximize sum of radii => minimize negative sum
    return -np.sum(r) + penalty


def _grid_init(w, h):
    """Generate initial grid positions for circles in rectangle w x h."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            cr = 0.5 * min(w, h) / side * 0.95
            cells.append((cx, cy, cr))
    return np.array(cells[:N])


def _optimize_for_aspect(w, h, deadline, best_val, best_circles):
    """Optimize circle positions and radii for a given aspect ratio."""
    # Grid initialization
    init = _grid_init(w, h)
    x0 = init[:, 0].copy()
    y0 = init[:, 1].copy()
    r0 = init[:, 2].copy()
    
    # Bounds for each variable
    bounds = []
    for i in range(N):
        bounds.append((0, w))  # x
    for i in range(N):
        bounds.append((0, h))  # y
    for i in range(N):
        bounds.append((1e-6, min(w, h) / 2))  # r
    
    v0 = np.concatenate([x0, y0, r0])
    
    # Try multiple starts with perturbation
    best_local = -np.inf
    best_v = v0.copy()
    
    n_starts = 5
    for s in range(n_starts):
        if _time_remaining(deadline) < 5:
            break
        
        if s == 0:
            v_start = v0.copy()
        else:
            v_start = v0 + np.random.normal(0, 0.05 * min(w, h), (3 * N,))
            # Clip to bounds
            v_start[:N] = np.clip(v_start[:N], 0, w)
            v_start[N:2*N] = np.clip(v_start[N:2*N], 0, h)
            v_start[2*N:3*N] = np.clip(v_start[2*N:3*N], 1e-6, min(w, h) / 2)
        
        try:
            res = minimize(_objective, v_start, args=(w, h), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12})
            if res.success or res.fun < best_local:
                best_local = res.fun
                best_v = res.x.copy()
        except Exception:
            continue
    
    if _time_remaining(deadline) < 2:
        return best_val, best_circles
    
    # Extract and validate
    x_opt = best_v[:N]
    y_opt = best_v[N:2*N]
    r_opt = best_v[2*N:3*N]
    
    if _validate(x_opt, y_opt, r_opt, w, h):
        val = np.sum(r_opt)
        if val > best_val:
            best_val = val
            best_circles = np.column_stack([x_opt, y_opt, r_opt])
    
    return best_val, best_circles


def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    best_val = 0.0
    best_circles = None
    
    # Fallback grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    fallback = np.array(cells[:N])
    if _validate(fallback[:, 0], fallback[:, 1], fallback[:, 2], 1.0, 1.0):
        best_val = np.sum(fallback[:, 2])
        best_circles = fallback.copy()
    
    # Try various aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.7, 1.3, 0.9, 1.1, 1.4, 0.75]
    
    for ar in aspect_ratios:
        if _time_remaining(deadline) < 10:
            break
        
        w = 1.0 * ar / (ar + 1.0) * 2.0
        h = 2.0 - w
        
        # Clamp to reasonable range
        w = np.clip(w, 0.4, 1.6)
        h = 2.0 - w
        
        best_val, best_circles = _optimize_for_aspect(w, h, deadline, best_val, best_circles)
    
    if best_circles is None:
        return fallback
    
    return best_circles
# EVOLVE-BLOCK-END
