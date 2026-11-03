import time
import math
import numpy as np
from scipy.optimize import minimize

N = 26
_DEADLINE = 150.0
_MARGIN = 2.0

def _greedy_init(n, rng):
    """
    Greedy circle packing initialization.
    Places n circles one by one, choosing the largest possible radius at each step.
    """
    circles = []
    grid_size = 40
    candidates = []
    for i in range(grid_size):
        for j in range(grid_size):
            candidates.append(((i + 0.5) / grid_size, (j + 0.5) / grid_size))
    for _ in range(500):
        candidates.append((rng.uniform(0, 1), rng.uniform(0, 1)))
    candidates = np.array(candidates)
    
    for _ in range(n):
        best_x, best_y, best_r = -1, -1, -1
        for cx, cy in candidates:
            r_boundary = min(cx, cy, 1 - cx, 1 - cy)
            if r_boundary <= 0:
                continue
            r_overlap = r_boundary
            for x, y, r in circles:
                dist = math.hypot(cx - x, cy - y)
                r_possible = dist - r
                if r_possible < r_overlap:
                    r_overlap = r_possible
            if r_overlap > best_r:
                best_r = r_overlap
                best_x = cx
                best_y = cy
        if best_r > 0:
            circles.append((best_x, best_y, best_r))
        else:
            circles.append((rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.01))
    pos = np.array([(c[0], c[1]) for c in circles])
    r = np.array([c[2] for c in circles])
    return pos, r

def _objective(params, beta):
    r = params[:N]
    pos = params[N:].reshape(N, 2)
    x = pos[:, 0]
    y = pos[:, 1]
    
    score = np.sum(r)
    
    b = np.maximum(np.maximum(r - x, r - y), np.maximum(r - (1.0 - x), r - (1.0 - y)))
    if np.any(b > 0):
        score -= beta * np.sum(b * b)
        
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    need = (r[:, None] + r[None, :]) - d
    np.fill_diagonal(need, 0.0)
    if np.any(need > 0):
        score -= beta * np.sum(need * need)
        
    return score

def _objective_and_grad(params, beta):
    r = params[:N]
    pos = params[N:].reshape(N, 2)
    x = pos[:, 0]
    y = pos[:, 1]
    
    grad_r = np.full(N, 1.0)
    grad_pos = np.zeros((N, 2))
    
    for i in range(N):
        r_i = r[i]
        xi, yi = x[i], y[i]
        
        v = r_i - xi
        if v > 0:
            grad_r[i] += -2.0 * beta * v
            grad_pos[i, 0] += 2.0 * beta * v
        v = r_i - yi
        if v > 0:
            grad_r[i] += -2.0 * beta * v
            grad_pos[i, 1] += 2.0 * beta * v
        v = r_i - (1.0 - xi)
        if v > 0:
            grad_r[i] += -2.0 * beta * v
            grad_pos[i, 0] += -2.0 * beta * v
        v = r_i - (1.0 - yi)
        if v > 0:
            grad_r[i] += -2.0 * beta * v
            grad_pos[i, 1] += -2.0 * beta * v
            
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    d_safe = np.where(d == 0, 1e-8, d)
    need = (r[:, None] + r[None, :]) - d
    for i in range(N):
        for j in range(i + 1, N):
            v = need[i, j]
            if v > 0:
                ux = (x[i] - x[j]) / d_safe[i, j]
                uy = (y[i] - y[j]) / d_safe[i, j]
                g = 2.0 * beta * v
                grad_r[i] += -g
                grad_r[j] += -g
                grad_pos[i, 0] += g * ux
                grad_pos[i, 1] += g * uy
                grad_pos[j, 0] += -g * ux
                grad_pos[j, 1] += -g * uy
                
    grad = np.concatenate([grad_r, grad_pos.ravel()])
    return _objective(params, beta), grad

def _is_valid(pos, r, tol=1e-7):
    x = pos[:, 0]
    y = pos[:, 1]
    if np.any(r < -tol) or np.any(x < -tol) or np.any(y < -tol) or \
       np.any(x > 1 + tol) or np.any(y > 1 + tol):
        return False
    if np.any(r + x < -tol) or np.any(r + y < -tol) or \
       np.any(r - (1 - x) < -tol) or np.any(r - (1 - y) < -tol):
        return False
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    need = (r[:, None] + r[None, :]) - d
    np.fill_diagonal(need, 0.0)
    if np.any(need > tol):
        return False
    return True

def construct():
    t0 = time.time()
    best_score = -np.inf
    best_result = None
    
    side = int(math.ceil(math.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    baseline = np.array(cells[:N])
    if _is_valid(baseline[:, :2], baseline[:, 2]):
        best_score = np.sum(baseline[:, 2])
        best_result = baseline.copy()
        
    rng = np.random.RandomState(42)
    
    n_starts = 15
    for s in range(n_starts):
        if time.time() > _DEADLINE - _MARGIN:
            break
            
        pos0, r0 = _greedy_init(N, rng)
        pos0 = pos0 + rng.uniform(-0.01, 0.01, pos0.shape)
        pos0 = np.clip(pos0, 0.0, 1.0)
        
        params = np.concatenate([r0, pos0.ravel()])
        bounds = [(0.0, 0.5) for _ in range(N)] + [(0.0, 1.0)] * (2 * N)
        
        start_beta = 1.0
        end_beta = 100000.0
        n_sweeps = 3
        max_iter = 200
        
        for sweep in range(n_sweeps):
            if time.time() > _DEADLINE - _MARGIN:
                break
                
            beta = start_beta * (end_beta / start_beta) ** (sweep / max(1, n_sweeps - 1))
            
            try:
                res = minimize(
                    lambda p, b=beta: _objective(p, b),
                    params,
                    jac=lambda p, b=beta: _objective_and_grad(p, b)[1],
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": max_iter, "maxfun": max_iter * 4, "ftol": 1e-12, "gtol": 1e-6},
                )
                params = res.x
            except Exception:
                pass
            
            r = params[:N]
            pos = params[N:].reshape(N, 2)
            
            if _is_valid(pos, r):
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([pos[:, 0], pos[:, 1], r])
                    
    return best_result
