import random
import time

def construct():
    # Target n values to try, starting from high
    target_ns = [28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
    
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    best_n_found = 0
    best_colors_found = None
    
    for n in target_ns:
        if time.time() - start_time > time_limit:
            break
            
        # Time budget for this n
        # Give more time to higher n
        remaining = time_limit - (time.time() - start_time)
        if n >= 27:
            time_budget = min(remaining, 40.0)
        elif n == 26:
            time_budget = min(remaining, 50.0)
        else:
            time_budget = min(remaining, 10.0)
            
        if time_budget < 2.0:
            continue
            
        end_time_for_n = time.time() + time_budget
        
        # We will try multiple restarts for high n
        # Each restart runs SA for a fraction of the budget
        # Or just one long run with many steps.
        # Let's stick to one long run but with more steps.
        
        # Initialize random coloring
        colors = [random.randint(0, 4) for _ in range(n * (n - 1) // 2)]
        
        # Build adjacency sets for each color
        adj = [ [set() for _ in range(n)] for _ in range(5) ]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
                idx += 1
                
        def count_c4s_in_color(c):
            init_count = 0
            for u in range(n):
                for v in adj[c][u]:
                    if u < v:
                        count = 0
                        for x in adj[c][v]:
                            if x == u: continue
                            for y in adj[c][u]:
                                if y == v: continue
                                if x == y: continue
                                if y in adj[c][x]:
                                    count += 1
                        init_count += count
            return init_count // 4

        total_energy = 0
        for c in range(5):
            total_energy += count_c4s_in_color(c)
            
        best_energy = total_energy
        best_colors = colors.copy()
        
        T = 100.0
        cooling = 0.999
        min_T = 0.01
        
        current_colors = colors.copy()
        
        def get_idx(i, j):
            if i > j: i, j = j, i
            return i * n - i * (i + 1) // 2 + (j - i - 1)
            
        steps = 0
        # Set max_steps high enough that time limit is the constraint
        max_steps = 2000000 
        
        def edge_c4_contribution(c, i, j):
            cnt = 0
            # Optimization: iterate over smaller set
            if len(adj[c][i]) < len(adj[c][j]):
                small = adj[c][i]
                large = adj[c][j]
                small_node = i
                large_node = j
            else:
                small = adj[c][j]
                large = adj[c][i]
                small_node = j
                large_node = i
                
            # We need x in N(j), y in N(i)
            # If we swap, we need to be careful.
            # Original: x in adj[c][j], y in adj[c][i]
            # Let's just stick to original to avoid bugs, deg is small.
            for x in adj[c][j]:
                if x == i: continue
                for y in adj[c][i]:
                    if y == j: continue
                    if x == y: continue
                    if y in adj[c][x]:
                        cnt += 1
            return cnt

        while T > min_T and steps < max_steps:
            if time.time() > end_time_for_n:
                break
                
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j: continue
            
            idx_ij = get_idx(i, j)
            c_old = current_colors[idx_ij]
            c_new = random.randint(0, 4)
            if c_old == c_new: continue
            
            delta_old = -edge_c4_contribution(c_old, i, j)
            delta_new = edge_c4_contribution(c_new, i, j)
            delta_E = delta_old + delta_new
            
            if delta_E < 0 or random.random() < pow(2.718, -delta_E / T):
                current_colors[idx_ij] = c_new
                adj[c_old][i].remove(j)
                adj[c_old][j].remove(i)
                adj[c_new][i].add(j)
                adj[c_new][j].add(i)
                
                total_energy += delta_E
                
                if total_energy < best_energy:
                    best_energy = total_energy
                    best_colors = current_colors.copy()
                    if best_energy == 0:
                        return {'n': n, 'colors': best_colors}
                        
            T *= cooling
            steps += 1
            
        if best_energy == 0:
            return {'n': n, 'colors': best_colors}
            
        # If we didn't find 0, but found a lower energy, we don't save it for lower n
        # because lower n is a different problem.
        # But we can update best_n_found if we found a valid one? 
        # No, the loop returns immediately on valid.
        # If we finish the loop for this n without valid, we go to next n.
        
    # Fallback
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
