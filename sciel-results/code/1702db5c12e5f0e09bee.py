import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s margin (90s kill)
    
    n = 128
    # Precompute all 4-term APs (nonconstant)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    # Map index to list of AP indices containing it
    idx_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for idx in ap:
            idx_to_aps[idx].append(i)
    
    def count_bad_colors(colors):
        """Count APs with <= 2 distinct colors."""
        bad = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) <= 2:
                bad += 1
        return bad
    
    best_solution = None
    best_cost = float('inf')
    
    # Try k = 3, 4, 5...
    for k in range(3, 10):
        if time.time() > deadline - 15:
            break
            
        # Local search for this k
        best_k_solution = None
        best_k_cost = float('inf')
        
        # Multiple restarts
        max_restarts = 80
        restart_count = 0
        while time.time() < deadline - 10 and restart_count < max_restarts:
            restart_count += 1
            
            # Random initialization with some structure
            if restart_count % 3 == 0:
                # Try a structured initialization
                colors = np.zeros(n, dtype=int)
                for i in range(n):
                    colors[i] = i % k
            else:
                colors = np.random.randint(0, k, n)
            
            current_cost = count_bad_colors(colors)
            
            if current_cost == 0:
                best_k_solution = colors.copy()
                best_k_cost = 0
                break
                
            # Simulated Annealing with better cooling
            T = 2.0
            T_min = 0.05
            cooling = 0.9995
            no_improve = 0
            max_iters = 15000
            
            for _ in range(max_iters):
                if time.time() > deadline - 10:
                    break
                    
                if current_cost == 0:
                    best_k_solution = colors.copy()
                    best_k_cost = 0
                    break
                    
                # Pick random index
                i = np.random.randint(n)
                old_color = colors[i]
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                    
                # Calculate delta cost efficiently
                delta = 0
                for ap_idx in idx_to_aps[i]:
                    ap = aps[ap_idx]
                    c = [colors[j] for j in ap]
                    pos_in_ap = ap.index(i)
                    c[pos_in_ap] = new_color
                    new_bad = 1 if len(set(c)) <= 2 else 0
                    old_bad = 1 if len(set([colors[j] for j in ap])) <= 2 else 0
                    delta += new_bad - old_bad
                    
                if delta < 0:
                    colors[i] = new_color
                    current_cost += delta
                    no_improve = 0
                else:
                    # Accept with probability exp(-delta/T)
                    if random.random() < np.exp(-delta / T):
                        colors[i] = new_color
                        current_cost += delta
                    else:
                        no_improve += 1
                        if no_improve > 200:
                            break
                            
                # Cool down
                T *= cooling
                if T < T_min:
                    T = T_min

            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()
                
            if best_k_cost == 0:
                break

        if best_k_cost == 0:
            # Found a valid solution with k colors
            return list(best_k_solution)
            
    # Fallback: Greedy construction
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            for d in range(1, (x-1)//3+1):
                if len({color} | {colors[x-1-j*d] for j in (1,2,3)}) < 3:
                    valid = False
                    break
            if not valid:
                continue
            for d in range(1, (x-1)//2+1):
                if all(colors[x-1-j*d] == color for j in (1,2)):
                    valid = False
                    break
            if not valid:
                continue
            colors.append(color)
            break
    return colors
