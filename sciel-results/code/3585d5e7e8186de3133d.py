"""Circle packing in perimeter-4 rectangle: Physical simulation + Simulated Annealing + Aspect Ratio Optimization."""
import numpy as np
import time

N = 21


def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """Check if packing is valid: all circles inside rectangle and non-overlapping."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(cx < r - tol) or np.any(cx > W - r + tol):
        return False
    if np.any(cy < r - tol) or np.any(cy > H - r + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d2 = dx * dx + dy * dy
            rs = r[i] + r[j]
            if d2 < rs * rs - tol:
                return False
    return True


def _total_radius(cx, cy, r):
    return float(np.sum(r))


def _simulate(cx, cy, r, W, H, n_steps=2000, dt=0.002, temp=0.01, seed=None):
    """
    Physical simulation with repulsive forces and temperature-controlled random kicks.
    Returns improved (cx, cy, r) after simulation.
    """
    if seed is not None:
        np.random.seed(seed)
    
    cx = cx.copy().astype(float)
    cy = cy.copy().astype(float)
    r = r.copy().astype(float)
    
    iu = np.triu_indices(N, k=1)
    
    for step in range(n_steps):
        temp_cur = temp * (0.999 ** step)
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        left_viol = np.maximum(r - cx, 0.0)
        right_viol = np.maximum(cx + r - W, 0.0)
        bottom_viol = np.maximum(r - cy, 0.0)
        top_viol = np.maximum(cy + r - H, 0.0)
        
        fx -= left_viol * 50.0
        fx += right_viol * 50.0
        fy -= bottom_viol * 50.0
        fy += top_viol * 50.0
        
        dx = cx[iu[0]] - cx[iu[1]]
        dy = cy[iu[0]] - cy[iu[1]]
        d2 = dx * dx + dy * dy
        d = np.sqrt(np.maximum(d2, 1e-12))
        r_sum = r[iu[0]] + r[iu[1]]
        overlap = np.maximum(r_sum - d, 0.0)
        
        if np.any(overlap > 0):
            force_mag = overlap * 20.0
            ux = dx / d
            uy = dy / d
            
            for k in range(len(iu[0])):
                i_idx = iu[0][k]
                j_idx = iu[1][k]
                fx[i_idx] += ux[k] * force_mag[k]
                fy[i_idx] += uy[k] * force_mag[k]
                fx[j_idx] -= ux[k] * force_mag[k]
                fy[j_idx] -= uy[k] * force_mag[k]
        
        if temp_cur > 1e-6:
            fx += np.random.randn(N) * temp_cur
            fy += np.random.randn(N) * temp_cur
        
        cx += dt * fx
        cy += dt * fy
        
        cx = np.clip(cx, r, W - r)
        cy = np.clip(cy, r, H - r)
    
    return cx, cy, r


def _grow_radii(cx, cy, r, W, H, max_rounds=50):
    """Try to grow all radii uniformly, then individually."""
    for _ in range(max_rounds):
        r_new = r * 1.001
        if _is_valid(cx, cy, r_new, W, H):
            r = r_new
        else:
            break
    
    for _ in range(10):
        improved = False
        for i in range(N):
            r_max_wall = min(cx[i], W - cx[i], cy[i], H - cy[i])
            r_max = r_max_wall
            for j in range(N):
                if i == j:
                    continue
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(r[i] + 1e-6, r_max)
            
            if r_max > r[i] + 1e-7:
                lo, hi = r[i], r_max
                for _ in range(15):
                    mid = (lo + hi) / 2
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(cx, cy, r_test, W, H):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-7:
                    r[i] = lo
                    improved = True
        if not improved:
            break
    
    return r


def _local_search(cx, cy, r, W, H, time_budget):
    """Local search: try moving each circle in small steps to improve total radius."""
    cx = cx.copy().astype(float)
    cy = cy.copy().astype(float)
    r = r.copy().astype(float)
    
    start_time = time.time()
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]
    
    for _ in range(100):
        if time.time() > time_budget:
            break
        
        improved = False
        for i in range(N):
            if time.time() > time_budget:
                break
            
            r_max_wall = min(cx[i], W - cx[i], cy[i], H - cy[i])
            r_max = r_max_wall
            for j in range(N):
                if i == j:
                    continue
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0, r_max)
            
            if r_max > r[i] + 1e-6:
                lo, hi = r[i], r_max
                for _ in range(12):
                    mid = (lo + hi) / 2
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(cx, cy, r_test, W, H):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-7:
                    r[i] = lo
                    improved = True
                    continue
            
            for dx_dir, dy_dir in directions:
                for step in [0.02, 0.01, 0.005, 0.002]:
                    new_x = cx[i] + dx_dir * step
                    new_y = cy[i] + dy_dir * step
                    if new_x < r[i] or new_x > W - r[i]:
                        continue
                    if new_y < r[i] or new_y > H - r[i]:
                        continue
                    
                    cx_test = cx.copy()
                    cy_test = cy.copy()
                    cx_test[i] = new_x
                    cy_test[i] = new_y
                    
                    if _is_valid(cx_test, cy_test, r, W, H):
                        cx[i] = new_x
                        cy[i] = new_y
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    
    return cx, cy, r


def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_value = 0.0
    best_result = None
    
    rng = np.random.default_rng(12345)
    
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.55, 1.8, 0.5]
    
    for trial in range(15):
        if time.time() > deadline - 30:
            break
        
        if trial < len(aspect_ratios):
            w = aspect_ratios[trial]
        else:
            w = rng.uniform(0.5, 1.5)
        h = 2.0 - w
        if h < 0.3:
            continue
        
        side = int(np.ceil(np.sqrt(N)))
        cx = np.zeros(N)
        cy = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            row, col = divmod(i, side)
            cx[i] = (col + 0.5) / side * w
            cy[i] = (row + 0.5) / side * h
            r[i] = min(w / side, h / side) * 0.4
        
        cx += rng.uniform(-0.02, 0.02, N)
        cy += rng.uniform(-0.02, 0.02, N)
        cx = np.clip(cx, r, w - r)
        cy = np.clip(cy, r, h - r)
        
        cx_sim, cy_sim, r_sim = _simulate(cx, cy, r, w, h, n_steps=1500, dt=0.002, temp=0.01, seed=trial)
        
        r_sim = _grow_radii(cx_sim, cy_sim, r_sim, w, h, max_rounds=30)
        
        cx_ls, cy_ls, r_ls = _local_search(cx_sim, cy_sim, r_sim, w, h, time_budget=deadline - 10)
        
        r_ls = _grow_radii(cx_ls, cy_ls, r_ls, w, h, max_rounds=20)
        
        val = _total_radius(cx_ls, cy_ls, r_ls)
        if val > best_value:
            best_value = val
            best_result = (cx_ls.copy(), cy_ls.copy(), r_ls.copy(), w, h)
    
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        cx = np.array([(i % side + 0.5) / side for i in range(N)])
        cy = np.array([(i // side + 0.5) / side for i in range(N)])
        r = np.full(N, 0.5 / side - 0.01)
        return np.column_stack((cx, cy, r))
    
    cx_b, cy_b, r_b, w_b, h_b = best_result
    
    if time.time() < deadline - 5:
        cx_p, cy_p, r_p = _local_search(cx_b, cy_b, r_b, w_b, h_b, deadline - 2)
        r_p = _grow_radii(cx_p, cy_p, r_p, w_b, h_b, max_rounds=10)
        val_p = _total_radius(cx_p, cy_p, r_p)
        if val_p > best_value:
            best_value = val_p
            best_result = (cx_p.copy(), cy_p.copy(), r_p.copy(), w_b, h_b)
    
    cx_b, cy_b, r_b, w_b, h_b = best_result
    
    for _ in range(5):
        r_trial = r_b * (1 - 1e-8)
        if _is_valid(cx_b, cy_b, r_trial, w_b, h_b):
            r_b = r_trial
        else:
            break
    
    return np.column_stack((cx_b, cy_b, r_b))
