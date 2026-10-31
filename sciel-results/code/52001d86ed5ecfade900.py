"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Annealed softplus-penalty L-BFGS-B with diverse structured inits + SLSQP polish.
- Objective: -sum(r_i) + w * sum(softplus(penetrations))
- Anneal w from small to large across restarts
- Many diverse initializations (hex, grid, corners, random, jittered)
- Final SLSQP polish with hard constraints
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 26
# EVOLVE-BLOCK-START

def _is_valid(circles, tol=1e-7):
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1.0 + tol:
            return False
        if y - r < -tol or y + r > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - tol:
                return False
    return True


def _post_process(circles, tol=1e-9):
    out = []
    for x, y, r in circles:
        r = max(0.0, r - tol)
        x = min(1.0 - r, max(r, x))
        y = min(1.0 - r, max(r, y))
        out.append((x, y, r))
    return out


def _sum_radii(circles):
    return sum(c[2] for c in circles)


def _softplus(x, beta=80.0):
    """Numerically stable softplus."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0
    out[pos] = x[pos] + np.log1p(np.exp(-beta * x[pos])) / beta
    out[~pos] = np.log1p(np.exp(beta * x[~pos])) / beta
    return out


def _softplus_grad(x, beta=80.0):
    """Gradient of softplus."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-beta * x))


def _penalty_objective(params, w, beta=80.0):
    """-sum(r) + w * sum(softplus(penetrations))."""
    n = N
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    obj = -np.sum(r)
    
    # Boundary penetrations
    pen_b = np.concatenate([
        r - x,          # left
        r - (1.0 - x),  # right
        r - y,          # bottom
        r - (1.0 - y),  # top
    ])
    obj += w * np.sum(_softplus(pen_b, beta))
    
    # Pairwise penetrations
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            pen = r[i] + r[j] - dist
            obj += w * _softplus(np.array([pen]), beta)[0]
    
    return obj


def _penalty_grad(params, w, beta=80.0):
    """Gradient of penalty objective."""
    n = N
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = np.zeros(n)
    
    # Objective gradient: -1 for each r
    gr -= 1.0
    
    # Boundary penalties
    # left: softplus(r - x)
    for i in range(n):
        pen = r[i] - x[i]
        g = _softplus_grad(np.array([pen]), beta)[0]
        gr[i] += w * g
        gx[i] -= w * g
        
        pen = r[i] - (1.0 - x[i])
        g = _softplus_grad(np.array([pen]), beta)[0]
        gr[i] += w * g
        gx[i] += w * g
        
        pen = r[i] - y[i]
        g = _softplus_grad(np.array([pen]), beta)[0]
        gr[i] += w * g
        gy[i] -= w * g
        
        pen = r[i] - (1.0 - y[i])
        g = _softplus_grad(np.array([pen]), beta)[0]
        gr[i] += w * g
        gy[i] += w * g
    
    # Pairwise penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-12:
                continue
            pen = r[i] + r[j] - dist
            g = _softplus_grad(np.array([pen]), beta)[0]
            if g > 1e-10:
                gr[i] += w * g
                gr[j] += w * g
                gx[i] += w * g * dx / dist
                gx[j] -= w * g * dx / dist
                gy[i] += w * g * dy / dist
                gy[j] -= w * g * dy / dist
    
    grad = np.zeros(3 * n)
    grad[0::3] = gx
    grad[1::3] = gy
    grad[2::3] = gr
    return grad


def _lbfgs_optimize(params, w, beta=80.0, max_iter=300):
    """Run L-BFGS-B with penalty objective."""
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.001, 0.5)] * N
    try:
        result = minimize(
            lambda p: _penalty_objective(p, w, beta),
            params,
            jac=lambda p: _penalty_grad(p, w, beta),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        return result.x
    except Exception:
        return params


def _slsqp_polish(circles):
    """SLSQP polish with hard inequality constraints."""
    params = np.array([val for c in circles for val in c])
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.001, 0.5)] * N
    
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i+1] - p[3*i+2]})
    
    for i in range(N):
        for j in range(i + 1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2
            })
    
    def objective(p):
        return -sum(p[3*i+2] for i in range(N))
    
    try:
        result = minimize(
            objective, params,
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        if result.success:
            return [(result.x[3*i], result.x[3*i+1], result.x[3*i+2]) for i in range(N)]
    except Exception:
        pass
    return circles


def _init_hexagonal(jitter=0.0):
    circles = []
    r = 0.075
    rows = [5, 5, 5, 5, 6]
    row_spacing = r * np.sqrt(3)
    col_spacing = 2 * r
    row_idx = 0
    for count in rows:
        offset = 0 if row_idx % 2 == 0 else col_spacing / 2
        total_width = col_spacing * (count - 1) + offset
        start_x = (1.0 - total_width) / 2.0
        y = r + row_spacing * row_idx + 0.03
        for col in range(count):
            x = start_x + offset + col * col_spacing
            if jitter > 0:
                x += random.uniform(-jitter, jitter)
                y += random.uniform(-jitter, jitter)
            if len(circles) < N:
                circles.append((x, y, r))
        row_idx += 1
    circles = circles[:N]
    while len(circles) < N:
        circles.append((0.5, 0.5, 0.01))
    return circles


def _init_grid():
    side = int(np.ceil(np.sqrt(N)))
    circles = []
    for k in range(N):
        i = k % side
        j = k // side
        x = (i + 0.5) / side
        y = (j + 0.5) / side
        r = 0.4 / side
        circles.append((x, y, r))
    return circles


def _init_corners():
    circles = []
    circles.append((0.08, 0.08, 0.08))
    circles.append((0.92, 0.08, 0.08))
    circles.append((0.08, 0.92, 0.08))
    circles.append((0.92, 0.92, 0.08))
    circles.append((0.5, 0.05, 0.06))
    circles.append((0.5, 0.95, 0.06))
    circles.append((0.05, 0.5, 0.06))
    circles.append((0.95, 0.5, 0.06))
    random.seed(42)
    while len(circles) < N:
        x = random.uniform(0.15, 0.85)
        y = random.uniform(0.15, 0.85)
        r = random.uniform(0.04, 0.07)
        circles.append((x, y, r))
    return circles


def _init_random(seed):
    random.seed(seed)
    circles = []
    for _ in range(N):
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        r = random.uniform(0.03, 0.08)
        circles.append((x, y, r))
    return circles


def _circles_to_params(circles):
    return np.array([val for c in circles for val in c])


def _params_to_circles(params):
    return [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Diverse initializations
    inits = [
        _init_hexagonal(0.0),
        _init_hexagonal(0.02),
        _init_hexagonal(0.04),
        _init_grid(),
        _init_corners(),
        _init_random(42),
        _init_random(123),
        _init_random(777),
        _init_random(999),
        _init_random(314),
        _init_random(271),
    ]
    
    best_circles = inits[0]
    best_sum = _sum_radii(best_circles)
    
    # Penalty weights to anneal through (increasing = harder constraints)
    weight_schedule = [1.0, 5.0, 20.0, 100.0, 500.0, 2000.0, 10000.0]
    
    # For each init, run annealed L-BFGS-B
    for init_idx, init_circles in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        
        params = _circles_to_params(init_circles)
        
        for w in weight_schedule:
            if time.time() > deadline - 20.0:
                break
            params = _lbfgs_optimize(params, w, beta=80.0, max_iter=200)
            circles = _params_to_circles(params)
            
            if _is_valid(circles, tol=1e-6):
                s = _sum_radii(circles)
                if s > best_sum:
                    best_sum = s
                    best_circles = circles
        
        # If still not valid, try with lower beta (smoother)
        if not _is_valid(_params_to_circles(params), tol=1e-6):
            params2 = _circles_to_params(init_circles)
            for w in [500.0, 5000.0]:
                if time.time() > deadline - 20.0:
                    break
                params2 = _lbfgs_optimize(params2, w, beta=40.0, max_iter=300)
                circles2 = _params_to_circles(params2)
                if _is_valid(circles2, tol=1e-6):
                    s = _sum_radii(circles2)
                    if s > best_sum:
                        best_sum = s
                        best_circles = circles2
    
    # SLSQP polish on best
    if time.time() < deadline - 10.0:
        polished = _slsqp_polish(best_circles)
        if _is_valid(polished, tol=1e-7):
            ps = _sum_radii(polished)
            if ps > best_sum:
                best_sum = ps
                best_circles = polished
    
    # One more round: small random perturbation + L-BFGS
    if time.time() < deadline - 8.0:
        random.seed(int(time.time()) % 10000)
        perturbed = []
        for x, y, r in best_circles:
            x = np.clip(x + random.uniform(-0.01, 0.01), 0.01, 0.99)
            y = np.clip(y + random.uniform(-0.01, 0.01), 0.01, 0.99)
            perturbed.append((x, y, r))
        p = _circles_to_params(perturbed)
        for w in [100.0, 1000.0]:
            if time.time() > deadline - 8.0:
                break
            p = _lbfgs_optimize(p, w, beta=80.0, max_iter=150)
        pc = _params_to_circles(p)
        if _is_valid(pc, tol=1e-6):
            ps = _sum_radii(pc)
            if ps > best_sum:
                best_sum = ps
                best_circles = pc
    
    # Final SLSQP
    if time.time() < deadline - 5.0:
        polished2 = _slsqp_polish(best_circles)
        if _is_valid(polished2, tol=1e-7):
            ps = _sum_radii(polished2)
            if ps > best_sum:
                best_sum = ps
                best_circles = polished2
    
    best_circles = _post_process(best_circles)
    
    if not _is_valid(best_circles):
        best_circles = _post_process(_init_grid())
    
    return np.array(best_circles)

# EVOLVE-BLOCK-END
