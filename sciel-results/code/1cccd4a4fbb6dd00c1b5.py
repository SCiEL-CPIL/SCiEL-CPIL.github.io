import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 26
    # ... initialization ...
    
    best_solution = None
    
    # Try multiple restarts
    for restart in range(100):
        # Init
        colors = [[-1] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                colors[i][j] = random.randint(0, 4)
                colors[j][i] = colors[i][j]
        
        neighbors = [[set() for _ in range(n)] for _ in range(5)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[i][j]
                neighbors[c][i].add(j)
                neighbors[c][j].add(i)
        
        c4_counts = [0] * 5
        for c in range(5):
             # Recalculate or use incremental? 
             # For initialization, just calculate once.
             total = 0
             for u in range(n):
                 for v in range(u + 1, n):
                     common = neighbors[c][u].intersection(neighbors[c][v])
                     k = len(common)
                     if k >= 2:
                         total += k * (k - 1) // 2
             c4_counts[c] = total // 2
             
        total_c4s = sum(c4_counts)
        
        temp = 1.0
        steps = 0
        max_steps = 50000
        stuck_counter = 0
        best_local_c4s = total_c4s
        
        while steps < max_steps and time.time() < deadline:
            if total_c4s == 0:
                # Found it
                res_colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        res_colors.append(colors[i][j])
                return {'n': 26, 'colors': res_colors}
                
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j: continue
            if i > j: i, j = j, i
            
            c_old = colors[i][j]
            c_new = random.randint(0, 4)
            if c_old == c_new: continue
            
            delta_remove = 0
            for x in neighbors[c_old][i]:
                if x == j: continue
                for y in neighbors[c_old][j]:
                    if y == i: continue
                    if x != y and y in neighbors[c_old][x]:
                        delta_remove += 1
                        
            delta_add = 0
            for x in neighbors[c_new][i]:
                for y in neighbors[c_new][j]:
                    if x != y and y in neighbors[c_new][x]:
                        delta_add += 1
                        
            delta_total = delta_add - delta_remove
            
            if delta_total <= 0 or random.random() < pow(random.random(), 1.0 / temp):
                # Update
                neighbors[c_old][i].remove(j)
                neighbors[c_old][j].remove(i)
                neighbors[c_new][i].add(j)
                neighbors[c_new][j].add(i)
                
                c4_counts[c_old] -= delta_remove
                c4_counts[c_new] += delta_add
                total_c4s += delta_total
                
                colors[i][j] = c_new
                colors[j][i] = c_new
                
                if total_c4s < best_local_c4s:
                    best_local_c4s = total_c4s
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                    
            temp *= 0.999
            if temp < 0.01: temp = 0.01
            steps += 1
            
        # If we didn't find 0 C4s, restart
        if time.time() > deadline - 5.0:
            break

    # Fallback to seed
    n_seed = 11
    return {'n': n_seed, 'colors': [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]}
