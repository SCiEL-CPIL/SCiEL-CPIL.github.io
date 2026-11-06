import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Prioritize higher n
    for n in [27, 26, 25, 24]:
        if time.time() > deadline - 10.0:
            break
        
        # Allocate time based on n
        # Give more time to 27, less to 26 if 27 fails quickly?
        # Let's give 20s to 27, 20s to 26.
        time_budget = 20.0
        if n == 27:
            time_budget = 30.0
        elif n == 26:
            time_budget = 30.0
            
        result = solve(n, time_budget)
        if result is not None:
            return result
            
    # Fallback
    return {'n': 11, 'colors': [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}

def solve(n, time_limit):
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute edge indices for fast lookup if needed, 
    # but for SA we just index the list directly.
    
    def calc_violations(colors):
        # Build neighbor bitmasks
        neighbors = [[0] * n for _ in range(5)]
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            neighbors[c][i] |= (1 << j)
            neighbors[c][j] |= (1 << i)
            
        violations = 0
        for c in range(5):
            for u in range(n):
                nu = neighbors[c][u]
                for v in range(u + 1, n):
                    nv = neighbors[c][v]
                    common = nu & nv
                    if common:
                        k = common.bit_count()
                        if k >= 2:
                            violations += k * (k - 1) // 2
        return violations

    # Initial random coloring
    colors = [random.randint(0, 4) for _ in range(num_edges)]
    current_violations = calc_violations(colors)
    
    if current_violations == 0:
        return {'n': n, 'colors': colors}
        
    best_violations = current_violations
    best_colors = colors[:]
    
    start_time = time.time()
    temperature = 1.0
    cooling_rate = 0.9999 # Slower cooling for more exploration
    min_temp = 0.01
    
    no_improve_steps = 0
    restart_threshold = 5000
    
    while time.time() < start_time + time_limit:
        # Pick random edge
        e_idx = random.randint(0, num_edges - 1)
        old_color = colors[e_idx]
        new_color = random.randint(0, 4)
        
        if new_color == old_color:
            continue
            
        colors[e_idx] = new_color
        new_violations = calc_violations(colors)
        
        delta = new_violations - current_violations
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors[:]
                no_improve_steps = 0
                if best_violations == 0:
                    return {'n': n, 'colors': colors}
            else:
                no_improve_steps += 1
        else:
            colors[e_idx] = old_color
            no_improve_steps += 1
            
        temperature *= cooling_rate
        if temperature < min_temp:
            temperature = min_temp
            
        # Restart mechanism
        if no_improve_steps > restart_threshold:
            # Reinitialize from best found so far with some noise
            colors = best_colors[:]
            current_violations = best_violations
            temperature = 1.0
            no_improve_steps = 0
            # Perturb
            for _ in range(10):
                e_idx = random.randint(0, num_edges - 1)
                colors[e_idx] = random.randint(0, 4)
            current_violations = calc_violations(colors)
            
    return None
