"""Circle packing in perimeter-4 rectangle: gradient-based optimizer with multi-start."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 140.0  # safety margin before 150s kill

def _time_ok():
    return time.time() < DEADLINE - 3.0

def _validity_check(cx, cy, r, W, H, tol=1e-8):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    for i in range(N):
        if r[i] < -tol:
            return False
        if cx[i] - r[i] < -tol or cx[i] + r[i] > W + tol:
            return False
        if cy[i] - r[i] < -tol or cy[i] + r[i] > H + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d < r[i] + r[j] - tol:
                return False
    return True

def _sum_radii(cx, cy, r, W, H):
    """Return sum of radii if valid, else 0."""
    if not _validity_check(cx, cy, r, W, H):
        return 0.0
    return float(np.sum(r))

def _objective(x, W, H, mu=1e4):
    """
    Objective for SLSQP: minimize -sum(r) + mu * penalty.
    x is shape (3N,): [cx_0..cx_N-1, cy_0..cy_N-1, r_0..r_N-1]
    """
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:]
    
    val = -np.sum(r)
    
    # Penalty for constraint violations
    pen = 0.0
    for i in range(N):
        # Boundary constraints: cx[i] - r[i] >= 0, cx[i] + r[i] <= W, etc.
        pen += max(0, -(cx[i] - r[i])) ** 2
        pen += max(0, (cx[i] + r[i]) - W) ** 2
        pen += max(0, -(cy[i] - r[i])) ** 2
        pen += max(0, (cy[i] + r[i]) - H) ** 2
        # Radius must be positive
        pen += max(0, -r[i]) ** 2
    
    # Overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            pen += max(0, r[i] + r[j] - d) ** 2
    
    return val + mu * pen

def _objective_grad(x, W, H, mu=1e4):
    """
    Analytic gradient of the objective.
    """
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:]
    
    grad = np.zeros(3 * N)
    
    # Gradient of -sum(r)
    grad[2*N:] = -1.0
    
    # Gradient of boundary penalties
    for i in range(N):
        # cx[i] - r[i] >= 0 -> penalty = max(0, -(cx[i]-r[i]))^2
        v = cx[i] - r[i]
        if v < 0:
            grad[i] += 2 * mu * (-v) * 1.0   # d/dcx of (-(cx-r))^2 = -2*(cx-r)*(-1)... 
            # Actually: f = (-(cx-r))^2 = (r-cx)^2, df/dcx = -2(r-cx) = 2(cx-r)
            # Wait: f = (r-cx)^2, df/dcx = 2(r-cx)*(-1) = -2(r-cx) = 2(cx-r)
            # But we have max(0, v)^2 where v = cx-r. If v<0, f=(v)^2, df/dv=2v, df/dcx=2v, df/dr=-2v
            grad[i] += 2 * mu * v * 1.0
            grad[2*N + i] += 2 * mu * v * (-1.0)
        
        v = cx[i] + r[i] - W
        if v > 0:
            grad[i] += 2 * mu * v * 1.0
            grad[2*N + i] += 2 * mu * v * 1.0
        
        v = cy[i] - r[i]
        if v < 0:
            grad[N + i] += 2 * mu * v * 1.0
            grad[2*N + i] += 2 * mu * v * (-1.0)
        
        v = cy[i] + r[i] - H
        if v > 0:
            grad[N + i] += 2 * mu * v * 1.0
            grad[2*N + i] += 2 * mu * v * 1.0
        
        # r >= 0
        if r[i] < 0:
            grad[2*N + i] += 2 * mu * r[i] * 1.0
    
    # Gradient of overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.hypot(dx, dy)
            if d < 1e-12:
                continue
            v = r[i] + r[j] - d
            if v > 0:
                # f = v^2, df/d(cx[i]) = 2v * (-dx/d), df/d(cy[i]) = 2v * (-dy/d)
                # df/d(cx[j]) = 2v * (dx/d), df/d(cy[j]) = 2v * (dy/d)
                # df/d(r[i]) = 2v, df/d(r[j]) = 2v
                factor = 2 * mu * v
                grad[i] += factor * (-dx / d)
                grad[N + i] += factor * (-dy / d)
                grad[j] += factor * (dx / d)
                grad[N + j] += factor * (dy / d)
                grad[2*N + i] += factor
                grad[2*N + j] += factor
    
    return grad

def _make_grid(kx, ky, W, H):
    """Create a kx by ky grid of circles, take first N."""
    pts = []
    for j in range(ky):
        for i in range(kx):
            cx = (i + 0.5) * W / kx
            cy = (j + 0.5) * H / ky
            r = min(cx, cy, W - cx, H - cy) * 0.95
            r = max(r, 1e-4)
            pts.append([cx, cy, r])
    arr = np.array(pts)
    if len(arr) < N:
        for _ in range(N - len(arr)):
            arr = np.vstack([arr, [0.05, 0.05, 0.01]])
    return arr[:N]

def _hexagonal_layout(W, H):
    """Create a hexagonal packing layout."""
    pts = []
    base_r = min(W, H) / 5.0
    row_h = base_r * np.sqrt(3)
    n_rows = int(H / row_h) + 2
    n_cols = int(W / (2 * base_r)) + 2
    
    for j in range(n_rows):
        y = j * row_h + base_r
        if y - base_r < 0 or y + base_r > H:
            continue
        offset = (j % 2) * base_r
        for i in range(n_cols):
            x = i * 2 * base_r + base_r + offset
            if x - base_r < 0 or x + base_r > W:
                continue
            r = base_r * 0.95
            r = min(r, x, W - x, y, H - y)
            if r > 1e-4:
                pts.append([x, y, r])
    
    if len(pts) == 0:
        return _make_grid(5, 5, W, H)
    
    arr = np.array(pts)
    order = np.argsort(-arr[:, 2])[:N]
    arr = arr[order]
    
    if len(arr) < N:
        for _ in range(N - len(arr)):
            arr = np.vstack([arr, [0.05, 0.05, 0.01]])
    return arr[:N]

def _random_layout(W, H):
    """Create a random valid-ish layout."""
    cx = np.random.uniform(0.1, W * 0.9, N)
    cy = np.random.uniform(0.1, H * 0.9, N)
    r = np.ones(N) * min(W, H) / 8.0
    for i in range(N):
        r[i] = min(r[i], cx[i], W - cx[i], cy[i], H - cy[i])
        r[i] = max(r[i], 1e-4)
    return np.column_stack([cx, cy, r])

def _optimize_one(W, H, x0, maxiter=200):
    """Optimize a single layout for given W, H using SLSQP."""
    x0 = x0.flatten().copy()
    
    # Clamp initial values
    x0[:N] = np.clip(x0[:N], 0, W)
    x0[N:2*N] = np.clip(x0[N:2*N], 0, H)
    x0[2*N:] = np.maximum(x0[2*N:], 1e-4)
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, W))  # cx
    for i in range(N):
        bounds.append((0, H))  # cy
    for i in range(N):
        bounds.append((1e-6, min(W, H)))  # r
    
    # Constraints
    cons = []
    for i in range(N):
        # cx[i] - r[i] >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*N+i], 'jac': lambda x, i=i: np.concatenate([np.eye(N)[i], np.zeros(N), np.eye(N)[i] * (-1)])})
        # cx[i] + r[i] <= W
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: W - x[i] - x[2*N+i], 'jac': lambda x, i=i: np.concatenate([np.eye(N)[i] * (-1), np.zeros(N), np.eye(N)[i] * (-1)])})
        # cy[i] - r[i] >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N+i] - x[2*N+i], 'jac': lambda x, i=i: np.concatenate([np.zeros(N), np.eye(N)[i], np.eye(N)[i] * (-1)])})
        # cy[i] + r[i] <= H
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: H - x[N+i] - x[2*N+i], 'jac': lambda x, i=i: np.concatenate([np.zeros(N), np.eye(N)[i] * (-1), np.eye(N)[i] * (-1)])})
    
    # Overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: np.hypot(x[i]-x[j], x[N+i]-x[N+j]) - x[2*N+i] - x[2*N+j]})
    
    try:
        res = minimize(
            _objective, x0, args=(W, H, 1e4),
            jac=_objective_grad,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-8}
        )
    except Exception:
        return None, 0.0
    
    if not _time_ok():
        return None, 0.0
    
    x = res.x
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:]
    
    # Clamp to ensure validity
    cx = np.clip(cx, 0, W)
    cy = np.clip(cy, 0, H)
    r = np.maximum(r, 1e-6)
    for i in range(N):
        r[i] = min(r[i], cx[i], W - cx[i], cy[i], H - cy[i])
    
    score = _sum_radii(cx, cy, r, W, H)
    if score <= 0:
        return None, 0.0
    
    out = np.column_stack([cx, cy, r])
    return out, score

def _relax(cx, cy, r, W, H, n_iter=50):
    """
    Iteratively try to increase each circle's radius given positions.
    """
    cx = cx.copy()
    cy = cy.copy()
    r = r.copy()
    
    for it in range(n_iter):
        improved = False
        for i in range(N):
            r_bound = min(cx[i], W - cx[i], cy[i], H - cy[i])
            r_max = r_bound
            for j in range(N):
                if i == j:
                    continue
                d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
                r_candidate = d - r[j]
                if r_candidate < r_max:
                    r_max = r_candidate
            r_max = max(r_max, 1e-6)
            if r_max > r[i] + 1e-10:
                r[i] = r_max
                improved = True
        if not improved:
            break
    
    return cx, cy, r

def construct():
    best_score = 0.0
    best_out = None
    best_W = 1.0
    
    if not _time_ok():
        W, H = 1.0, 1.0
        g = _make_grid(5, 5, W, H)
        cx, cy, r = g[:, 0], g[:, 1], g[:, 2]
        cx, cy, r = _relax(cx, cy, r, W, H, n_iter=20)
        score = _sum_radii(cx, cy, r, W, H)
        if score > 0:
            return np.column_stack([cx, cy, r])
        return g
    
    # Try various aspect ratios
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    for W in aspect_ratios:
        if not _time_ok():
            break
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        # Generate multiple starting layouts
        candidates = []
        
        # Grid layouts
        for kx, ky in [(4, 5), (5, 4), (3, 7), (7, 3), (5, 5), (4, 6), (6, 4)]:
            if kx * ky >= N:
                candidates.append(_make_grid(kx, ky, W, H))
        
        # Hexagonal
        candidates.append(_hexagonal_layout(W, H))
        
        # Random
        for _ in range(2):
            candidates.append(_random_layout(W, H))
        
        # Optimize each candidate
        for g in candidates:
            if not _time_ok():
                break
            out, score = _optimize_one(W, H, g, maxiter=150)
            if out is not None and score > best_score:
                best_score = score
                best_out = out
                best_W = W
        
        # Perturbation restarts on best
        if best_out is not None and _time_ok():
            for _ in range(2):
                if not _time_ok():
                    break
                noise = np.random.randn(*best_out.shape) * 0.02
                perturbed = best_out + noise
                perturbed[:, 0] = np.clip(perturbed[:, 0], 0, W)
                perturbed[:, 1] = np.clip(perturbed[:, 1], 0, H)
                perturbed[:, 2] = np.maximum(perturbed[:, 2], 1e-4)
                out, score = _optimize_one(W, H, perturbed, maxiter=100)
                if out is not None and score > best_score:
                    best_score = score
                    best_out = out
                    best_W = W
    
    # Focused refinement on best aspect ratio
    if best_out is not None and _time_ok():
        W = best_W
        H = 2.0 - W
        for _ in range(3):
            if not _time_ok():
                break
            noise = np.random.randn(*best_out.shape) * 0.01
            perturbed = best_out + noise
            perturbed[:, 0] = np.clip(perturbed[:, 0], 0, W)
            perturbed[:, 1] = np.clip(perturbed[:, 1], 0, H)
            perturbed[:, 2] = np.maximum(perturbed[:, 2], 1e-4)
            out, score = _optimize_one(W, H, perturbed, maxiter=300)
            if out is not None and score > best_score:
                best_score = score
                best_out = out
    
    # Final relaxation pass
    if best_out is not None and _time_ok():
        W = best_W
        H = 2.0 - W
        cx, cy, r = best_out[:, 0], best_out[:, 1], best_out[:, 2]
        cx, cy, r = _relax(cx, cy, r, W, H, n_iter=100)
        score = _sum_radii(cx, cy, r, W, H)
        if score > best_score:
            best_score = score
            best_out = np.column_stack([cx, cy, r])
    
    # Fallback
    if best_out is None:
        W, H = 1.0, 1.0
        g = _make_grid(5, 5, W, H)
        cx, cy, r = g[:, 0], g[:, 1], g[:, 2]
        cx, cy, r = _relax(cx, cy, r, W, H, n_iter=30)
        if _sum_radii(cx, cy, r, W, H) > 0:
            best_out = np.column_stack([cx, cy, r])
        else:
            best_out = g
    
    return best_out
