"""Circle packing in perimeter-4 rectangle: DE for topology + SLSQP for refinement."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
START = time.time()
DEADLINE = START + 135.0  # Safety margin before 150s kill

def _max_feasible_radii(centers, W, H):
    """
    Given centers and box dimensions, compute the maximum radius for each circle
    such that all circles are disjoint and inside the box.
    Returns an array of radii.
    """
    n = len(centers)
    radii = np.zeros(n)
    for i in range(n):
        cx, cy = centers[i]
        # Distance to walls
        r = min(cx, W - cx, cy, H - cy)
        if r <= 0:
            return None # Invalid center
        
        # Distance to other circles
        for j in range(n):
            if i == j:
                continue
            dx = cx - centers[j][0]
            dy = cy - centers[j][1]
            d = np.hypot(dx, dy)
            # To be disjoint, r_i + r_j <= d
            # If we fix r_j, max r_i is d - r_j.
            # But we want max r_i assuming r_j is also maximized?
            # Actually, if we just want the max radius for circle i *given* other circles 
            # are at their positions with *some* radius, it's tricky.
            # The DE objective in inspiration 1 calculates:
            # r_i = min( wall_dist, min_j( (d_ij - r_j) ) )
            # This is iterative. 
            # However, a simpler and robust objective for "tightness" is:
            # r_i = min( wall_dist, min_j( d_ij / 2.0 ) )
            # This assumes equal radii or just uses half distance as a proxy for 
            # "room". This is a common heuristic for packing.
            # Let's stick to the inspiration's logic which is:
            # r_i is limited by d_ij - r_j. But r_j depends on r_i?
            # No, the inspiration calculates r_i based on *current* r_j? 
            # Actually, inspiration 1 says: "maximize the sum of maximum feasible radii".
            # It computes r = min(wall, min_j(d/2)).
            # This is a valid lower bound on the max radius if we assume others are 0?
            # No, d/2 is the radius if the other circle has radius 0.
            # If the other circle has radius R, my max radius is d - R.
            # If we just use d/2, we are ignoring the size of the neighbor.
            # This might lead to overlapping if we don't enforce it later.
            # But SLSQP will fix the radii.
            # Let's use a stricter bound: r_i <= (d_ij - r_j) is circular.
            # Let's use the objective: sum of min(wall_dist, min_j(d_ij / 2)).
            # This encourages centers to be far apart.
            r = min(r, d / 2.0)
            if r <= 0:
                return None
        radii[i] = r
    return radii

def _de_objective(x):
    """
    Objective for Differential Evolution.
    x[0] = W (width), x[1:] = centers (2N values)
    Returns negative sum of max feasible radii (using d/2 heuristic).
    """
    W = x[0]
    H = 2.0 - W
    if W < 0.2 or H < 0.2:
        return 1e6
    
    centers = x[1:].reshape(N, 2)
    
    # Quick check for centers inside box
    if np.any(centers < 0) or np.any(centers[:, 0] > W) or np.any(centers[:, 1] > H):
        return 1e6
        
    radii = _max_feasible_radii(centers, W, H)
    if radii is None:
        return 1e6
        
    return -np.sum(radii)

def _slsqp_objective(x, W, H, mu=1e3):
    """
    Objective for SLSQP.
    x is shape (N,3): cx, cy, r.
    Returns (value, grad).
    Value = -sum(r) + penalties for violations.
    """
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    n = len(cx)

    # Clip for numerical stability in gradient calculation
    cx_c = np.clip(cx, 0.0, W)
    cy_c = np.clip(cy, 0.0, H)
    r_c = np.clip(r, 1e-6, 1.0)

    val = -np.sum(r_c)
    grad = np.zeros_like(x)
    grad[:, 2] = -1.0

    # Boundary constraints: r <= min(cx, cy, W-cx, H-cy)
    lo = np.minimum(np.minimum(cx_c, cy_c), np.minimum(W - cx_c, H - cy_c))
    viol_b = np.maximum(r_c - lo, 0.0)
    val += mu * np.sum(viol_b ** 2)
    
    for i in range(n):
        if viol_b[i] > 0:
            d_left = cx_c[i]
            d_bottom = cy_c[i]
            d_right = W - cx_c[i]
            d_top = H - cy_c[i]
            min_d = lo[i]
            eps = 1e-9
            g_cx = 0.0
            g_cy = 0.0
            cnt = 0
            if d_left <= min_d + eps:
                g_cx += 1.0; cnt += 1
            if d_right <= min_d + eps:
                g_cx += -1.0; cnt += 1
            if d_bottom <= min_d + eps:
                g_cy += 1.0; cnt += 1
            if d_top <= min_d + eps:
                g_cy += -1.0; cnt += 1
            if cnt > 0:
                g_cx /= cnt
                g_cy /= cnt
            dv = 2.0 * mu * viol_b[i]
            grad[i, 0] += dv * (-g_cx)
            grad[i, 1] += dv * (-g_cy)
            grad[i, 2] += dv * 1.0

    # Non-overlap: max(r_i + r_j - dist, 0)^2
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx_c[i] - cx_c[j]
            dy = cy_c[i] - cy_c[j]
            d = np.hypot(dx, dy)
            req = r_c[i] + r_c[j]
            viol = req - d
            if viol <= 0:
                continue
            val += mu * viol ** 2
            if d < 1e-12:
                grad[i, 0] += 2.0 * mu * viol * 1.0
                grad[j, 0] += 2.0 * mu * viol * (-1.0)
                grad[i, 2] += 2.0 * mu * viol * 1.0
                grad[j, 2] += 2.0 * mu * viol * 1.0
                continue
            dv = 2.0 * mu * viol
            grad[i, 0] += dv * (-dx / d)
            grad[i, 1] += dv * (-dy / d)
            grad[j, 0] += dv * (dx / d)
            grad[j, 1] += dv * (dy / d)
            grad[i, 2] += dv * 1.0
            grad[j, 2] += dv * 1.0

    return val, grad

def _evaluate(x, W, H):
    """Return sum of radii if valid, else None."""
    cx = np.clip(x[:, 0], 0.0, W)
    cy = np.clip(x[:, 1], 0.0, H)
    r = np.maximum(x[:, 2], 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(W - cx, H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(cx[i] - cx[j], cy[i] - cy[j])
            if d + 1e-8 < r[i] + r[j]:
                return None
    return float(np.sum(r))

def _grid_layout(kx, ky, W, H, shrink=0.999):
    """Build a kx*ky grid layout."""
    cx = np.linspace(0.0, W, kx + 2)[1:-1]
    cy = np.linspace(0.0, H, ky + 2)[1:-1]
    pts = []
    for yy in cy:
        for xx in cx:
            r = min(xx, yy, W - xx, H - yy) * shrink
            pts.append([xx, yy, max(r, 1e-4)])
    arr = np.array(pts)
    if len(arr) < N:
        while len(arr) < N:
            arr = np.vstack([arr, [0.01, 0.01, 0.005]])
    return arr[:N]

def _run_de(bounds, pop_size=30, max_iter=200, time_limit=60.0):
    """Simple Differential Evolution implementation."""
    n_dims = len(bounds)
    start_time = time.time()
    
    # Initialize population
    pop = np.zeros((pop_size, n_dims))
    for i in range(pop_size):
        for j in range(n_dims):
            pop[i, j] = random.uniform(bounds[j][0], bounds[j][1])
    
    fitness = np.array([_de_objective(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]
    
    for gen in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        new_pop = []
        new_fitness = []
        
        for i in range(pop_size):
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            F = 0.5 + random.random() * 0.5
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            for j in range(n_dims):
                mutant[j] = np.clip(mutant[j], bounds[j][0], bounds[j][1])
            
            trial = pop[i].copy()
            for j in range(n_dims):
                if random.random() < 0.7:
                    trial[j] = mutant[j]
            
            f_trial = _de_objective(trial)
            
            if f_trial <= fitness[i]:
                new_pop.append(trial)
                new_fitness.append(f_trial)
            else:
                new_pop.append(pop[i])
                new_fitness.append(fitness[i])
        
        pop = np.array(new_pop)
        fitness = np.array(new_fitness)
        
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_f:
            best_f = fitness[gen_best_idx]
            best_x = pop[gen_best_idx].copy()
            
    return best_x, best_f

def construct():
    best_sum = 0.0
    best_arr = None
    best_W = 1.0
    best_H = 1.0
    
    random.seed(42)
    np.random.seed(42)
    
    # Phase 1: Differential Evolution to find good topologies
    # Bounds: [W, cx1, cy1, ..., cxN, cyN]
    # W in [0.5, 1.5] (since H = 2-W, H in [0.5, 1.5])
    # Centers in [0, 2] (loose, will be clipped)
    bounds = [(0.5, 1.5)]
    for _ in range(N * 2):
        bounds.append((0.0, 2.0))
        
    # Run DE for a few seconds
    de_time_limit = 40.0
    de_start = time.time()
    
    try:
        de_best_x, de_best_f = _run_de(bounds, pop_size=20, max_iter=100, time_limit=de_time_limit)
    except Exception:
        de_best_x = None
        
    candidates = []
    
    # Add DE result as a candidate
    if de_best_x is not None:
        W = de_best_x[0]
        H = 2.0 - W
        centers = de_best_x[1:].reshape(N, 2)
        # Clamp centers
        centers[:, 0] = np.clip(centers[:, 0], 0.0, W)
        centers[:, 1] = np.clip(centers[:, 1], 0.0, H)
        
        # Estimate radii from DE objective (d/2 heuristic)
        radii = _max_feasible_radii(centers, W, H)
        if radii is not None:
            x0 = np.column_stack([centers, radii])
            s = _evaluate(x0, W, H)
            if s is not None:
                candidates.append((s, W, H, x0))
    
    # Add Grid Layouts as candidates (robust fallback and often good starting points)
    grid_sizes = [(3, 4), (4, 3), (3, 3), (4, 4), (5, 3), (3, 5)]
    Ws = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for W in Ws:
        H = 2.0 - W
        for kx, ky in grid_sizes:
            if kx * ky < N:
                continue
            g = _grid_layout(kx, ky, W, H, shrink=0.99)
            s = _evaluate(g, W, H)
            if s is not None:
                candidates.append((s, W, H, g))
                
    # Sort candidates by initial score
    candidates.sort(key=lambda t: -t[0])
    
    # Phase 2: Refine top candidates with SLSQP
    # We have ~90 seconds left.
    # Let's refine the top 5 candidates.
    top_k = min(5, len(candidates))
    
    for idx in range(top_k):
        if time.time() > DEADLINE - 10:
            break
            
        s0, W, H, g0 = candidates[idx]
        
        def fun(x):
            v, _ = _slsqp_objective(x, W, H)
            return v
            
        def grad(x):
            _, g = _slsqp_objective(x, W, H)
            return g
            
        try:
            res = minimize(fun, g0, jac=grad, method='SLSQP',
                           options={'maxiter': 100, 'ftol': 1e-10, 'disp': 0})
            if res.x is not None:
                s = _evaluate(res.x, W, H)
                if s is not None and s > best_sum:
                    best_sum = s
                    best_arr = res.x
                    best_W = W
                    best_H = H
        except Exception:
            pass
            
    # Fallback if no valid solution found
    if best_arr is None:
        W, H = 1.0, 1.0
        g = _grid_layout(5, 5, W, H, shrink=0.99)
        best_arr = g
        best_W, best_H = W, H
        
    # Final clamping
    cx = np.clip(best_arr[:, 0], 0.0, best_W)
    cy = np.clip(best_arr[:, 1], 0.0, best_H)
    r = best_arr[:, 2].copy()
    r = np.maximum(r, 1e-6)
    lo = np.minimum(np.minimum(cx, cy), np.minimum(best_W - cx, best_H - cy))
    r = np.minimum(r, lo)
    r = np.maximum(r, 1e-6)
    out = np.column_stack([cx, cy, r])
    
    # Final validation
    s_final = _evaluate(out, best_W, best_H)
    if s_final is None:
        # Fallback to grid
        W, H = 1.0, 1.0
        g = _grid_layout(5, 5, W, H, shrink=0.99)
        return g
        
    return out

