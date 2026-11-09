# EVOLVE-BLOCK-START
import numpy as np
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    max_time = 580.0
    
    # Function to calculate score and validity
    def check_and_score(centers, sides):
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return -np.inf, False
        if np.any(sides <= 0):
            return -np.inf, False
        if np.any(centers - sides/2 < -1e-9) or np.any(centers + sides/2 > 1 + 1e-9):
            return -np.inf, False
            
        cx = centers[:, 0]
        cy = centers[:, 1]
        
        dx = np.abs(cx[:, np.newaxis] - cx[np.newaxis, :])
        dy = np.abs(cy[:, np.newaxis] - cy[np.newaxis, :])
        s_sum = (sides[:, np.newaxis] + sides[np.newaxis, :]) / 2.0
        
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        
        dx = dx[mask]
        dy = dy[mask]
        s_sum = s_sum[mask]
        
        ox = np.maximum(0, s_sum - dx)
        oy = np.maximum(0, s_sum - dy)
        penalty = np.sum(ox * oy)
        
        score = np.sum(sides) - penalty * 100.0
        is_valid = penalty < 1e-9
        return score, is_valid

    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    # Multiple initializations
    inits = []
    
    # Init 1: 3x3 grid + 1 small
    s_grid = 1.0 / 3.0
    c1 = np.zeros((n, 2))
    s1 = np.zeros(n)
    idx = 0
    for i in range(3):
        for j in range(3):
            c1[idx, 0] = (j + 0.5) * s_grid
            c1[idx, 1] = (i + 0.5) * s_grid
            s1[idx] = s_grid
            idx += 1
    c1[idx, 0] = 0.5
    c1[idx, 1] = 0.5
    s1[idx] = 1e-6
    inits.append((c1, s1))
    
    # Init 2: Random
    rng = np.random.default_rng(123)
    for _ in range(5):
        c_r = rng.uniform(0.1, 0.9, size=(n, 2))
        s_r = rng.uniform(0.1, 0.3, size=n)
        # Check if valid
        _, is_valid = check_and_score(c_r, s_r)
        if is_valid:
            inits.append((c_r, s_r))
    
    for c_init, s_init in inits:
        if time.time() - start_time > max_time:
            break
            
        current_centers = c_init.copy()
        current_sides = s_init.copy()
        current_score, _ = check_and_score(current_centers, current_sides)
        
        step_size = 0.05
        temp = 0.05
        rng_search = np.random.default_rng(int(time.time() * 1000) % 10000)
        
        # Run local search for this init
        max_iters = 50000
        for _ in range(max_iters):
            if time.time() - start_time > max_time:
                break
                
            i = rng_search.integers(0, n)
            
            delta_c = rng_search.normal(0, step_size, size=2)
            new_cx = current_centers[i, 0] + delta_c[0]
            new_cy = current_centers[i, 1] + delta_c[1]
            
            delta_s = rng_search.normal(0, step_size / 2)
            new_s = current_sides[i] + delta_s
            if new_s <= 0:
                new_s = 1e-6
            
            old_cx = current_centers[i, 0]
            old_cy = current_centers[i, 1]
            old_s = current_sides[i]
            
            current_centers[i, 0] = new_cx
            current_centers[i, 1] = new_cy
            current_sides[i] = new_s
            
            new_score, is_valid = check_and_score(current_centers, current_sides)
            
            if new_score > current_score:
                current_score = new_score
                if is_valid and new_score > best_score:
                    best_centers = current_centers.copy()
                    best_sides = current_sides.copy()
                    best_score = new_score
            else:
                current_centers[i, 0] = old_cx
                current_centers[i, 1] = old_cy
                current_sides[i] = old_s
                
            temp *= 0.999
            if temp < 1e-5:
                temp = 1e-5

    # Fallback to baseline if no valid found (should not happen)
    if best_centers is None:
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(n, side, dtype=np.float64)

    return best_centers, best_sides

# EVOLVE-BLOCK-END
