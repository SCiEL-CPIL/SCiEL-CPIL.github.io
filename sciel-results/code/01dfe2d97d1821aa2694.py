# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
            return False
        if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if t > 0:
                if dx < t and dy < t:
                    return False
    return True

def _objective(centers, sides, penalty_weight=1e6):
    """
    Objective: sum(sides) - penalty. Higher is better.
    Uses smooth exponential penalty for overlaps.
    """
    n = len(sides)
    penalty = 0.0
    
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        # Smooth boundary penalty
        if cx - s/2 < 0:
            penalty += np.exp(-20 * (cx - s/2))
        if cx + s/2 > 1:
            penalty += np.exp(-20 * (1 - cx - s/2))
        if cy - s/2 < 0:
            penalty += np.exp(-20 * (cy - s/2))
        if cy + s/2 > 1:
            penalty += np.exp(-20 * (1 - cy - s/2))
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0
            if dx < t and dy < t:
                overlap_x = t - dx
                overlap_y = t - dy
                # Smooth penalty: use product with exponential scaling
                penalty += overlap_x * overlap_y * 50
    
    return np.sum(sides) - penalty_weight * penalty

def _scipy_optimize(centers, sides, deadline):
    """
    Uses scipy.optimize.minimize to refine the configuration.
    """
    n = len(sides)
    # Flatten to vector: [cx0, cy0, s0, cx1, cy1, s1, ...]
    x0 = np.concatenate([centers.flatten(), sides])
    
    def fun(x):
        # Check time
        if time.time() > deadline:
            return 1e10 # Large penalty to stop
        
        c = x[:2*n].reshape(n, 2)
        s = x[2*n:]
        
        # Clip to valid ranges to help optimizer
        s_clipped = np.clip(s, 0.01, 0.5)
        
        c_clipped = np.zeros_like(c)
        for i in range(n):
            c_clipped[i, 0] = np.clip(c[i, 0], s_clipped[i]/2, 1 - s_clipped[i]/2)
            c_clipped[i, 1] = np.clip(c[i, 1], s_clipped[i]/2, 1 - s_clipped[i]/2)
            
        # We want to maximize sum(s) - penalty
        # So minimize - (sum(s) - penalty)
        obj = _objective(c_clipped, s_clipped)
        return -obj

    # Use Nelder-Mead with more iterations for better convergence
    try:
        res = minimize(fun, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
    except Exception:
        return centers, sides

    x_opt = res.x
    c_opt = x_opt[:2*n].reshape(n, 2)
    s_opt = x_opt[2*n:]
    
    # Final clipping
    s_opt = np.clip(s_opt, 0.01, 0.5)
    for i in range(n):
        c_opt[i, 0] = np.clip(c_opt[i, 0], s_opt[i]/2, 1 - s_opt[i]/2)
        c_opt[i, 1] = np.clip(c_opt[i, 1], s_opt[i]/2, 1 - s_opt[i]/2)
        
    return c_opt, s_opt

def _local_search(centers, sides, max_iters=3000, deadline=None, rng_seed=42):
    """
    More aggressive local search with multiple strategies.
    """
    n = len(sides)
    best_score = np.sum(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    
    rng = np.random.RandomState(rng_seed)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Strategy 1: Try to grow a single square (primary strategy)
        i = rng.randint(n)
        for attempt in range(15):
            # Try to grow more aggressively
            grow = rng.random() < 0.8
            if grow:
                s_new = best_s[i] * (1 + rng.uniform(0.01, 0.12))
            else:
                s_new = best_s[i] * (1 + rng.uniform(-0.05, 0.05))
            s_new = max(0.01, min(0.5, s_new))
            
            # Try multiple positions
            cx_new = best_c[i, 0] + rng.uniform(-0.08, 0.08)
            cy_new = best_c[i, 1] + rng.uniform(-0.08, 0.08)
            
            cx_new = max(s_new/2, min(1 - s_new/2, cx_new))
            cy_new = max(s_new/2, min(1 - s_new/2, cy_new))
            
            c_new = best_c.copy()
            s_new_arr = best_s.copy()
            c_new[i] = [cx_new, cy_new]
            s_new_arr[i] = s_new
            
            if _is_valid(c_new, s_new_arr):
                score = np.sum(s_new_arr)
                if score > best_score + 1e-12:
                    best_score = score
                    best_c = c_new
                    best_s = s_new_arr
                    improved = True
                    break
        
        # Strategy 2: Perturb two squares simultaneously
        if not improved and rng.random() < 0.5:
            i1 = rng.randint(n)
            i2 = rng.randint(n)
            if i1 != i2:
                for attempt in range(6):
                    c_new = best_c.copy()
                    s_new_arr = best_s.copy()
                    for idx in [i1, i2]:
                        s_new = s_new_arr[idx] * (1 + rng.uniform(-0.03, 0.08))
                        s_new = max(0.01, min(0.5, s_new))
                        cx_new = c_new[idx, 0] + rng.uniform(-0.05, 0.05)
                        cy_new = c_new[idx, 1] + rng.uniform(-0.05, 0.05)
                        cx_new = max(s_new/2, min(1 - s_new/2, cx_new))
                        cy_new = max(s_new/2, min(1 - s_new/2, cy_new))
                        c_new[idx] = [cx_new, cy_new]
                        s_new_arr[idx] = s_new
                    
                    if _is_valid(c_new, s_new_arr):
                        score = np.sum(s_new_arr)
                        if score > best_score + 1e-12:
                            best_score = score
                            best_c = c_new
                            best_s = s_new_arr
                            improved = True
                            break
        
        # Strategy 3: Perturb three squares
        if not improved and rng.random() < 0.3:
            idxs = rng.choice(n, size=3, replace=False)
            for attempt in range(3):
                c_new = best_c.copy()
                s_new_arr = best_s.copy()
                for idx in idxs:
                    s_new = s_new_arr[idx] * (1 + rng.uniform(-0.02, 0.06))
                    s_new = max(0.01, min(0.5, s_new))
                    cx_new = c_new[idx, 0] + rng.uniform(-0.04, 0.04)
                    cy_new = c_new[idx, 1] + rng.uniform(-0.04, 0.04)
                    cx_new = max(s_new/2, min(1 - s_new/2, cx_new))
                    cy_new = max(s_new/2, min(1 - s_new/2, cy_new))
                    c_new[idx] = [cx_new, cy_new]
                    s_new_arr[idx] = s_new
                
                if _is_valid(c_new, s_new_arr):
                    score = np.sum(s_new_arr)
                    if score > best_score + 1e-12:
                        best_score = score
                        best_c = c_new
                        best_s = s_new_arr
                        improved = True
                        break
        
        # Strategy 4: Global random restart (rare)
        if not improved and rng.random() < 0.01:
            c_new = best_c.copy()
            s_new_arr = best_s.copy()
            for idx in range(n):
                s_new = s_new_arr[idx] * (1 + rng.uniform(-0.15, 0.15))
                s_new = max(0.01, min(0.5, s_new))
                cx_new = rng.uniform(s_new/2, 1 - s_new/2)
                cy_new = rng.uniform(s_new/2, 1 - s_new/2)
                c_new[idx] = [cx_new, cy_new]
                s_new_arr[idx] = s_new
            if _is_valid(c_new, s_new_arr):
                score = np.sum(s_new_arr)
                if score > best_score + 1e-12:
                    best_score = score
                    best_c = c_new
                    best_s = s_new_arr
    
    return best_c, best_s

def _coordinate_descent(centers, sides, max_iters=500, deadline=None):
    """
    Coordinate descent: for each square, optimize its position and size.
    """
    n = len(sides)
    best_score = np.sum(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        for i in range(n):
            best_local_score = np.sum(best_s)
            best_local_c = best_c.copy()
            best_local_s = best_s.copy()
            
            # Try to optimize size first
            for s_try in np.linspace(0.05, 0.5, 20):
                if s_try <= best_s[i] * 0.8 or s_try >= best_s[i] * 1.2:
                    continue
                # Try to find a valid position
                for _ in range(30):
                    cx = np.random.uniform(s_try/2, 1 - s_try/2)
                    cy = np.random.uniform(s_try/2, 1 - s_try/2)
                    
                    c_try = best_local_c.copy()
                    s_try_arr = best_local_s.copy()
                    c_try[i] = [cx, cy]
                    s_try_arr[i] = s_try
                    
                    if _is_valid(c_try, s_try_arr):
                        score = np.sum(s_try_arr)
                        if score > best_local_score:
                            best_local_score = score
                            best_local_c = c_try
                            best_local_s = s_try_arr
            
            # Now optimize position with fixed size
            s_fixed = best_local_s[i]
            for _ in range(20):
                cx = np.random.uniform(s_fixed/2, 1 - s_fixed/2)
                cy = np.random.uniform(s_fixed/2, 1 - s_fixed/2)
                
                c_try = best_local_c.copy()
                c_try[i] = [cx, cy]
                
                if _is_valid(c_try, best_local_s):
                    score = np.sum(best_local_s)
                    if score > best_local_score:
                        best_local_score = score
                        best_local_c = c_try
            
            best_c = best_local_c
            best_s = best_local_s
            best_score = best_local_score
    
    return best_c, best_s

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    # Fallback: 2x5 grid
    fallback_c = np.zeros((n, 2))
    fallback_s = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        fallback_c[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        fallback_s[i] = 0.2
    update_best(fallback_c, fallback_s)
    
    # Generate diverse initial configurations
    configs = []
    
    # 2x5 grid
    c1 = np.zeros((n, 2))
    s1 = np.zeros(n)
    for i in range(n):
        r = i // 5
        c = i % 5
        c1[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        s1[i] = 0.2
    configs.append((c1, s1))
    
    # 1x10 strip
    c2 = np.zeros((n, 2))
    s2 = np.zeros(n)
    for i in range(n):
        c2[i] = [(i + 0.5) * 0.1, 0.5]
        s2[i] = 0.1
    configs.append((c2, s2))
    
    # 5x2 grid
    c3 = np.zeros((n, 2))
    s3 = np.zeros(n)
    for i in range(n):
        r = i % 2
        c = i // 2
        c3[i] = [(c + 0.5) * 0.2, (r + 0.5) * 0.5]
        s3[i] = 0.2
    configs.append((c3, s3))
    
    # 3x3 grid + 1
    c4 = np.zeros((n, 2))
    s4 = np.zeros(n)
    for i in range(9):
        r = i // 3
        c = i % 3
        c4[i] = [(c + 0.5) / 3.0, (r + 0.5) / 3.0]
        s4[i] = 1.0/3.0
    c4[9] = [0.5, 0.5]
    s4[9] = 0.1
    configs.append((c4, s4))
    
    # Known good packing: mix of large and small squares
    # Try a configuration with 4 large squares in corners and 6 smaller ones
    c5 = np.zeros((n, 2))
    s5 = np.zeros(n)
    # 4 corner squares of size 0.25
    corners = [(0.125, 0.125), (0.875, 0.125), (0.125, 0.875), (0.875, 0.875)]
    for i, (cx, cy) in enumerate(corners):
        c5[i] = [cx, cy]
        s5[i] = 0.25
    # 4 edge squares of size 0.2
    edges = [(0.5, 0.1), (0.1, 0.5), (0.9, 0.5), (0.5, 0.9)]
    for i, (cx, cy) in enumerate(edges):
        c5[4+i] = [cx, cy]
        s5[4+i] = 0.2
    # 2 center squares of size 0.15
    c5[8] = [0.35, 0.35]
    s5[8] = 0.15
    c5[9] = [0.65, 0.65]
    s5[9] = 0.15
    if _is_valid(c5, s5):
        configs.append((c5, s5))
    
    # Another known good: 2 large + 8 small
    c6 = np.zeros((n, 2))
    s6 = np.zeros(n)
    c6[0] = [0.25, 0.25]
    s6[0] = 0.4
    c6[1] = [0.75, 0.75]
    s6[1] = 0.4
    # 8 small squares of size 0.12 in remaining space
    small_positions = [
        [0.25, 0.75], [0.75, 0.25], [0.5, 0.1], [0.5, 0.9],
        [0.1, 0.5], [0.9, 0.5], [0.5, 0.5], [0.3, 0.6]
    ]
    for i, pos in enumerate(small_positions):
        c6[2+i] = pos
        s6[2+i] = 0.12
    if _is_valid(c6, s6):
        configs.append((c6, s6))
    
    # Random initializations with greedy placement
    rng = np.random.RandomState(123)
    for trial in range(60):
        if time.time() > deadline:
            break
        
        s_rand = rng.uniform(0.08, 0.38, n)
        c_rand = np.zeros((n, 2))
        placed = np.zeros(n, dtype=bool)
        
        order = np.argsort(-s_rand)
        
        for idx in order:
            s = s_rand[idx]
            found = False
            for _ in range(400):
                cx = rng.uniform(s/2, 1 - s/2)
                cy = rng.uniform(s/2, 1 - s/2)
                c_rand[idx] = [cx, cy]
                
                ok = True
                for j in range(n):
                    if placed[j] and j != idx:
                        dx = abs(c_rand[idx, 0] - c_rand[j, 0])
                        dy = abs(c_rand[idx, 1] - c_rand[j, 1])
                        t = (s_rand[idx] + s_rand[j]) / 2.0
                        if dx < t and dy < t:
                            ok = False
                            break
                if ok:
                    placed[idx] = True
                    found = True
                    break
            
            if not found:
                s_rand[idx] = max(0.05, s_rand[idx] * 0.75)
        
        if _is_valid(c_rand, s_rand):
            configs.append((c_rand, s_rand))
    
    # For each config, run optimization
    for c_init, s_init in configs:
        if time.time() > deadline:
            break
        
        update_best(c_init, s_init)
        
        # 1. Run SciPy Optimizer (Stronger)
        if time.time() < deadline - 15:
            c_scipy, s_scipy = _scipy_optimize(c_init, s_init, deadline)
            update_best(c_scipy, s_scipy)
            
            # 2. Run Local Search on the result of SciPy (Polish)
            if time.time() < deadline - 10:
                c_ls, s_ls = _local_search(c_scipy, s_scipy, max_iters=300, deadline=deadline, rng_seed=42)
                update_best(c_ls, s_ls)
        
    # Final polish: try to squeeze more out of the best configuration
    if best_centers is not None and time.time() < deadline - 30:
        c_final, s_final = _scipy_optimize(best_centers, best_sides, deadline)
        update_best(c_final, s_final)
        
        if time.time() < deadline - 15:
            c_final2, s_final2 = _local_search(c_final, s_final, max_iters=800, deadline=deadline, rng_seed=777)
            update_best(c_final2, s_final2)
        
        # Additional polish with different random seed
        if time.time() < deadline - 5:
            c_final3, s_final3 = _local_search(c_final2, s_final2, max_iters=400, deadline=deadline, rng_seed=12345)
            update_best(c_final3, s_final3)
    
    if best_centers is None:
        best_centers = fallback_c
        best_sides = fallback_s
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
