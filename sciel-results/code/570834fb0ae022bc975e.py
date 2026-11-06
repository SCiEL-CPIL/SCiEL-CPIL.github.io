import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    N = 128
    
    # Precompute all non-constant 4-term arithmetic progressions
    aps = []
    index_to_aps = [[] for _ in range(N)]
    
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            ap = (i0, i1, i2, i3)
            aps.append(ap)
            index_to_aps[i0].append(ap)
            index_to_aps[i1].append(ap)
            index_to_aps[i2].append(ap)
            index_to_aps[i3].append(ap)
            
    # Helper to calculate cost for a single AP
    def ap_cost(colors, ap):
        c0 = colors[ap[0]]
        c1 = colors[ap[1]]
        c2 = colors[ap[2]]
        c3 = colors[ap[3]]
        return 0 if len({c0, c1, c2, c3}) >= 3 else 1

    # Helper to calculate full cost
    def calc_full_cost(colors):
        cost = 0
        for ap in aps:
            cost += ap_cost(colors, ap)
        return cost

    # 1. Generate a baseline valid solution using the greedy method
    colors_greedy = []
    for x in range(1, N + 1):
        found = False
        for color in range(128):
            valid = True
            max_d = (x - 2) // 3
            for d in range(1, max_d + 1):
                a_idx = x - 1 - 3 * d
                if a_idx < 0:
                    continue
                c1 = colors_greedy[a_idx]
                c2 = colors_greedy[a_idx + d]
                c3 = colors_greedy[a_idx + 2 * d]
                if len({c1, c2, c3, color}) < 3:
                    valid = False
                    break
            if valid:
                colors_greedy.append(color)
                found = True
                break
        if not found:
            colors_greedy.append(0)
            
    best_solution = colors_greedy
    best_num_colors = len(set(best_solution))
    
    # 2. Try to reduce the number of colors
    for K in range(3, best_num_colors):
        if time.time() > deadline:
            break
            
        # Run local search
        current_colors = np.random.randint(0, K, N)
        current_cost = calc_full_cost(current_colors)
        
        if current_cost == 0:
            actual_colors = len(set(current_colors))
            if actual_colors < best_num_colors:
                best_solution = current_colors.tolist()
                best_num_colors = actual_colors
            if K == 3:
                break
            continue

        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        k_start = time.time()
        k_deadline = k_start + 10.0
        
        while time.time() < k_deadline and T > min_T:
            # Pick a random index
            idx = np.random.randint(0, N)
            old_color = current_colors[idx]
            new_color = np.random.randint(0, K)
            
            if old_color == new_color:
                continue
            
            # Calculate delta cost by checking only APs involving idx
            delta = 0
            for ap in index_to_aps[idx]:
                delta -= ap_cost(current_colors, ap)
                current_colors[idx] = new_color
                delta += ap_cost(current_colors, ap)
                current_colors[idx] = old_color
            
            if delta < 0 or (delta > 0 and random.random() < np.exp(-delta / T)):
                current_colors[idx] = new_color
                current_cost += delta
                
                if current_cost == 0:
                    actual_colors = len(set(current_colors))
                    if actual_colors < best_num_colors:
                        best_solution = current_colors.tolist()
                        best_num_colors = actual_colors
                    break
            
            T *= cooling
            
        if current_cost == 0:
            actual_colors = len(set(current_colors))
            if actual_colors < best_num_colors:
                best_solution = current_colors.tolist()
                best_num_colors = actual_colors
            break

    return [int(c) for c in best_solution]
