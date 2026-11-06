import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Fallback: n=11 (known valid cyclic construction)
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback

    def is_valid_coloring(colors, n):
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return False
        
        adj_mask = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 4:
                    return False
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj_mask[c][u] & adj_mask[c][v]
                    if common.bit_count() > 1:
                        return False
        return True

    def count_violations(colors, n):
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return float('inf')
            
        adj_mask = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 4:
                    return float('inf')
                adj_mask[c][i] |= (1 << j)
                adj_mask[c][j] |= (1 << i)
                idx += 1
        
        violations = 0
        for c in range(5):
            for u in range(n):
                mask_u = adj_mask[c][u]
                for v in range(u + 1, n):
                    common = mask_u & adj_mask[c][v]
                    cnt = common.bit_count()
                    if cnt > 1:
                        violations += cnt - 1
        return violations

    def local_search(n, time_limit):
        num_edges = n * (n - 1) // 2
        best_colors = None
        best_violations = float('inf')
        
        # Multiple restarts
        max_restarts = 10
        for r in range(max_restarts):
            if time.time() > time_limit:
                break
            
            # Initialize randomly
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            current_violations = count_violations(colors, n)
            
            if current_violations == 0:
                return colors
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
            
            # Simulated Annealing
            temp = 5.0
            cooling = 0.999
            # Number of iterations per restart
            # Estimate time per iteration
            # ~0.01 ms per iteration?
            # Let's just run until time limit for this restart
            restart_deadline = time_limit - (time_limit - start_time) * 0.1 # keep 10% for future restarts?
            # Better: allocate time per restart
            time_per_restart = (time_limit - time.time()) / (max_restarts - r)
            restart_deadline = time.time() + time_per_restart
            
            for it in range(100000): # Cap iterations
                if time.time() > restart_deadline:
                    break
                if current_violations == 0:
                    return colors
                
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                new_color = random.randint(0, 4)
                if new_color == old_color:
                    continue
                
                colors[e_idx] = new_color
                new_violations = count_violations(colors, n)
                delta = new_violations - current_violations
                
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_violations = new_violations
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_colors = colors[:]
                else:
                    colors[e_idx] = old_color
                
                temp *= cooling
        
        return best_colors

    # Try to find valid coloring for n=26
    if time.time() < deadline - 10.0:
        n_target = 26
        colors_26 = local_search(n_target, deadline - 5.0)
        if colors_26 is not None and count_violations(colors_26, n_target) == 0:
            best_n = n_target
            best_colors = colors_26

    # If n=26 failed, try n=25, 24... down to best_n+1
    if best_n < 26:
        for n_try in range(25, best_n, -1):
            if time.time() > deadline - 5.0:
                break
            if n_try <= best_n:
                break
            colors_try = local_search(n_try, deadline - 5.0)
            if colors_try is not None and count_violations(colors_try, n_try) == 0:
                best_n = n_try
                best_colors = colors_try
                break # Found a valid one, stop searching smaller

    # Final validation
    if not is_valid_coloring(best_colors, best_n):
        best_n = n_fallback
        best_colors = colors_fallback

    return {'n': best_n, 'colors': best_colors}
