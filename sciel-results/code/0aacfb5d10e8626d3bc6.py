"""
Strategy: Multi-start SLSQP with constraint-based polish.
We use a unified penalty-based NLP approach:
1. Generate diverse initial configurations (hexagonal, grid, random).
2. For each, run SLSQP to maximize sum of radii subject to:
   - Circle-circle non-overlap
   - Circle-boundary containment
   - Perimeter constraint (W+H <= 2)
3. Polish the result with a tighter-tolerance pass.
The key structural change from the seed: instead of fixed-lattice scaling,
we jointly optimize positions, radii, AND container dimensions, letting
the optimizer find non-uniform radii and asymmetric packings that lattices
cannot achieve.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 21
DEADLINE_MARGIN = 15.0  # seconds before hard limit


def _vectorized_objective(params):
    """
    params layout: [x(N), y(N), r(N), W, H]
    Objective: maximize sum(r) => minimize -sum(r)
    Returns (value, grad) where value = -sum(r) + penalty
    """
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    W = params[3*N]
    H = params[3*N + 1]
    
    # Ensure positivity
    r = np.maximum(r, 1e-8)
    W = max(W, 1e-8)
    H = max(H, 1e-8)
    
    base_val = -np.sum(r)
    
    # Penalty for overlaps
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_d = r[i] + r[j]
            if dist_sq < min_d * min_d:
                dist = np.sqrt(dist_sq)
                overlap = min_d - dist
                penalty += overlap * overlap
    
    # Penalty for boundary violations
    for i in range(N):
        if x[i] < r[i]:
            penalty += (r[i] - x[i])**2
        if x[i] > W - r[i]:
            penalty += (x[i] - (W - r[i]))**2
        if y[i] < r[i]:
            penalty += (r[i] - y[i])**2
        if y[i] > H - r[i]:
            penalty += (y[i] - (H - r[i]))**2
    
    # Penalty for perimeter violation
    perim = 2.0*(W + H)
    if perim > 4.0:
        penalty += (perim - 4.0)**2
    
    return base_val + 100.0 * penalty


def _force_relax_unified(config, W, H, n_iters=200, dt=0.01):
    """
    Force-directed relaxation to resolve overlaps and boundary issues.
    config: (N,3) array of [x, y, r]
    """
    config = config.copy().astype(float)
    xs = config[:, 0].copy()
    ys = config[:, 1].copy()
    rs = config[:, 2].copy()
    
    for it in range(n_iters):
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Circle-circle repulsion
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_d = rs[i] + rs[j]
                if dist < min_d and dist > 1e-12:
                    overlap = min_d - dist
                    force = overlap / dist
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
                elif dist <= 1e-12:
                    # Jitter
                    fx[i] += 0.01 * random.uniform(-1, 1)
                    fy[i] += 0.01 * random.uniform(-1, 1)
        
        # Boundary forces
        for i in range(N):
            if xs[i] < rs[i]:
                fx[i] += (rs[i] - xs[i]) * 5.0
            if xs[i] > W - rs[i]:
                fx[i] -= (xs[i] - (W - rs[i])) * 5.0
            if ys[i] < rs[i]:
                fy[i] += (rs[i] - ys[i]) * 5.0
            if ys[i] > H - rs[i]:
                fy[i] -= (ys[i] - (H - rs[i])) * 5.0
        
        # Update
        xs += dt * fx
        ys += dt * fy
        
        # Clamp to boundaries
        for i in range(N):
            xs[i] = np.clip(xs[i], rs[i], W - rs[i])
            ys[i] = np.clip(ys[i], rs[i], H - rs[i])
        
        # Decay step size
        dt *= 0.995
    
    config[:, 0] = xs
    config[:, 1] = ys
    return config


def _polish_result(config, W, H, time_limit=30.0):
    """
    High-precision polish: run SLSQP with tight tolerances on the near-optimal config.
    """
    start = time.time()
    
    def objective(params):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        w = params[3*N]
        h = params[3*N + 1]
        
        val = -np.sum(r)
        
        # Hard penalty for any violation
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_d = r[i] + r[j]
                if dist_sq < min_d*min_d:
                    dist = np.sqrt(dist_sq)
                    pen += (min_d - dist)**2
        for i in range(N):
            if x[i] < r[i]: pen += (r[i] - x[i])**2
            if x[i] > w - r[i]: pen += (x[i] - (w - r[i]))**2
            if y[i] < r[i]: pen += (r[i] - y[i])**2
            if y[i] > h - r[i]: pen += (y[i] - (h - r[i]))**2
        if 2*(w+h) > 4.0:
            pen += (2*(w+h) - 4.0)**2
        
        return val + 1000.0 * pen
    
    x0 = np.concatenate([config[:, 0], config[:, 1], config[:, 2], [W, H]])
    
    # Bounds: radii positive, W,H positive
    bounds = []
    for i in range(N):
        bounds.append((0.0, None))  # x
    for i in range(N):
        bounds.append((0.0, None))  # y
    for i in range(N):
        bounds.append((1e-6, 1.0))  # r
    bounds.append((0.1, 2.0))  # W
    bounds.append((0.1, 2.0))  # H
    
    constraints = []
    
    # Perimeter constraint
    def perim_con(params):
        return 4.0 - 2.0*(params[3*N] + params[3*N+1])
    constraints.append({'type': 'ineq', 'fun': perim_con})
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
        if res.success or res.fun < objective(x0):
            new_config = np.column_stack([res.x[:N], res.x[N:2*N], res.x[2*N:3*N]])
            new_W = res.x[3*N]
            new_H = res.x[3*N+1]
            # Validate
            if _is_valid(new_config, new_W, new_H, tol=1e-6):
                return new_config, new_W, new_H
    except Exception:
        pass
    
    return config, W, H


def _constraint_refine(config, W, H):
    """
    Refine by adjusting radii upward where possible (greedy radius increase).
    """
    config = config.copy()
    improved = True
    max_rounds = 20
    
    for _ in range(max_rounds):
        if not improved:
            break
        improved = False
        for i in range(N):
            # Try increasing radius of circle i
            old_r = config[i, 2]
            # Find max possible radius
            max_r = min(W - config[i, 0], config[i, 0], H - config[i, 1], config[i, 1])
            # Check against other circles
            for j in range(N):
                if j == i:
                    continue
                dx = config[i, 0] - config[j, 0]
                dy = config[i, 1] - config[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - config[j, 2])
            
            if max_r > old_r + 1e-8:
                config[i, 2] = max_r - 1e-7
                improved = True
    
    return config


def _is_valid(config, W, H, tol=1e-5):
    """Check if configuration is valid."""
    if W <= 0 or H <= 0:
        return False
    if 2*(W + H) > 4.0 + tol:
        return False
    for i in range(N):
        x, y, r = config[i]
        if r < 0:
            return False
        if x < r - tol or x > W - r + tol:
            return False
        if y < r - tol or y > H - r + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            dx = config[i, 0] - config[j, 0]
            dy = config[i, 1] - config[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < config[i, 2] + config[j, 2] - tol:
                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_sum_r = 0.0
    best_config = None
    best_W = 1.0
    best_H = 1.0
    
    np.random.seed(42)
    random.seed(42)
    
    def try_packing(config, W, H):
        nonlocal best_sum_r, best_config, best_W, best_H
        if not _is_valid(config, W, H, tol=1e-4):
            # Try to fix
            config = _force_relax_unified(config, W, H, n_iters=100, dt=0.02)
            if not _is_valid(config, W, H, tol=1e-4):
                return
        sum_r = np.sum(config[:, 2])
        if sum_r > best_sum_r:
            best_sum_r = sum_r
            best_config = config.copy()
            best_W = W
            best_H = H
    
    # Generate diverse initial packings
    n_starts = 0
    max_starts = 12
    
    for attempt in range(max_starts):
        if time.time() > deadline - 40.0:
            break
        
        # Choose aspect ratio
        if attempt < 4:
            # Systematic aspect ratios
            ar = 0.7 + 0.15 * attempt  # W/H from 0.7 to 1.15
        else:
            ar = random.uniform(0.6, 1.4)
        
        H_init = 2.0 / (ar + 1.0)
        W_init = ar * H_init
        
        # Generate initial layout
        if attempt % 3 == 0:
            # Hexagonal
            config = _gen_hex(N, W_init, H_init)
        elif attempt % 3 == 1:
            # Grid
            config = _gen_grid(N, W_init, H_init)
        else:
            # Random with force relaxation
            config = _gen_random(N, W_init, H_init)
        
        if config is None:
            continue
        
        # Force relax to resolve any overlaps
        config = _force_relax_unified(config, W_init, H_init, n_iters=150, dt=0.015)
        
        # SLSQP optimization
        params = np.concatenate([config[:, 0], config[:, 1], config[:, 2], [W_init, H_init]])
        
        def obj(params):
            x = params[:N]
            y = params[N:2*N]
            r = np.maximum(params[2*N:3*N], 1e-8)
            w = max(params[3*N], 0.1)
            h = max(params[3*N+1], 0.1)
            val = -np.sum(r)
            pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i]-x[j]; dy = y[i]-y[j]
                    d2 = dx*dx+dy*dy
                    md = r[i]+r[j]
                    if d2 < md*md:
                        pen += (md - np.sqrt(d2))**2
            for i in range(N):
                if x[i] < r[i]: pen += (r[i]-x[i])**2
                if x[i] > w-r[i]: pen += (x[i]-(w-r[i]))**2
                if y[i] < r[i]: pen += (r[i]-y[i])**2
                if y[i] > h-r[i]: pen += (y[i]-(h-r[i]))**2
            if 2*(w+h) > 4.0:
                pen += (2*(w+h)-4.0)**2
            return val + 50.0*pen
        
        bounds = [(0.0, None)]*N + [(0.0, None)]*N + [(1e-6, 0.8)]*N + [(0.2, 2.0), (0.2, 2.0)]
        
        try:
            res = minimize(obj, params, method='SLSQP', bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-10})
            if res.fun < obj(params):
                new_cfg = np.column_stack([res.x[:N], res.x[N:2*N], np.maximum(res.x[2*N:3*N], 1e-6)])
                new_W = res.x[3*N]
                new_H = res.x[3*N+1]
                # Polish
                new_cfg, new_W, new_H = _polish_result(new_cfg, new_W, new_H, time_limit=20.0)
                # Constraint refine
                new_cfg = _constraint_refine(new_cfg, new_W, new_H)
                try_packing(new_cfg, new_W, new_H)
        except Exception:
            pass
        
        n_starts += 1
    
    # Final polish on best
    if best_config is not None and time.time() < deadline - 5.0:
        best_config, best_W, best_H = _polish_result(best_config, best_W, best_H, time_limit=15.0)
        best_config = _constraint_refine(best_config, best_W, best_H)
        if not _is_valid(best_config, best_W, best_H, tol=1e-4):
            best_config = _force_relax_unified(best_config, best_W, best_H, n_iters=200, dt=0.01)
    
    if best_config is not None:
        # Final validation and fix
        if not _is_valid(best_config, best_W, best_H, tol=1e-5):
            best_config = _force_relax_unified(best_config, best_W, best_H, n_iters=300, dt=0.005)
            # Shrink radii slightly if still invalid
            if not _is_valid(best_config, best_W, best_H, tol=1e-5):
                for i in range(N):
                    best_config[i, 2] *= 0.999
                best_config = _force_relax_unified(best_config, best_W, best_H, n_iters=100, dt=0.01)
        return best_config
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    W = 1.0
    H = 1.0
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) * W / side
            y = (j + 0.5) * H / side
            r = 0.5 * min(W, H) / side - 1e-4
            cells.append([x, y, r])
    return np.array(cells[:N])


def _gen_hex(n, W, H):
    """Generate hexagonal packing in W x H box."""
    config = np.zeros((n, 3))
    # Estimate radius
    r = min(W, H) / (2 * np.sqrt(n) * 1.1)
    
    # Hex grid
    dx = 2.0 * r
    dy = np.sqrt(3) * r
    pts = []
    row = 0
    while len(pts) < n:
        offset = 0.0 if row % 2 == 0 else r
        col = 0
        while len(pts) < n:
            x = offset + col * dx
            y = row * dy
            if x + r > W or y + r > H:
                break
            if x - r < 0 or y - r < 0:
                col += 1
                continue
            pts.append((x, y))
            col += 1
        row += 1
    
    if len(pts) < n:
        return None
    
    # Center in box
    xs = [p[0] for p in pts[:n]]
    ys = [p[1] for p in pts[:n]]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    shift_x = W/2.0 - cx
    shift_y = H/2.0 - cy
    
    for i in range(n):
        config[i, 0] = xs[i] + shift_x
        config[i, 1] = ys[i] + shift_y
        config[i, 2] = r
    
    return config


def _gen_grid(n, W, H):
    """Generate square grid packing in W x H box."""
    cols = int(np.ceil(np.sqrt(n * W / H)))
    rows = int(np.ceil(n / cols))
    
    r = min(W / cols, H / rows) / 2.0 - 0.001
    if r < 0.01:
        return None
    
    config = np.zeros((n, 3))
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 0.5) * W / cols
            y = (i + 0.5) * H / rows
            config[idx, 0] = x
            config[idx, 1] = y
            config[idx, 2] = r
            idx += 1
        if idx >= n:
            break
    
    if idx < n:
        return None
    return config


def _gen_random(n, W, H):
    """Generate random packing with force relaxation."""
    r = min(W, H) / (2 * np.sqrt(n) * 1.2)
    config = np.zeros((n, 3))
    for i in range(n):
        config[i, 0] = random.uniform(r, W - r)
        config[i, 1] = random.uniform(r, H - r)
        config[i, 2] = r
    return config


# EVOLVE-BLOCK-END
