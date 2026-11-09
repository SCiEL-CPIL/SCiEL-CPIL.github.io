import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 128
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    num_aps = len(aps)
    
    # Map index to list of AP indices containing it
    idx_to_aps = [[] for _ in range(n)]
    # For each AP, store the 4 indices
    for i, ap in enumerate(aps):
        for idx in ap:
            idx_to_aps[idx].append(i)
    
    # Precompute for each AP and each position in it, the other 3 indices
    # other3[ap_idx][pos_in_ap] = tuple of other 3 indices
    other3 = []
    for ap in aps:
        o = []
        for p in range(4):
            others = tuple(ap[j] for j in range(4) if j != p)
            o.append(others)
        other3.append(o)
    
    # pos_in_ap[ap_idx][idx] = position of idx in ap (0-3), or -1
    pos_in_ap = [[-1]*n for _ in range(num_aps)]
    for i, ap in enumerate(aps):
        for p, idx in enumerate(ap):
            pos_in_ap[i][idx] = p
    
    def count_bad(colors):
        """Count APs with <= 2 distinct colors."""
        bad = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) <= 2:
                bad += 1
        return bad
    
    def compute_delta(colors, i, old_color, new_color):
        """Compute cost delta when changing position i from old_color to new_color."""
        delta = 0
        for ap_idx in idx_to_aps[i]:
            ap = aps[ap_idx]
            p = pos_in_ap[ap_idx][i]
            others = other3[ap_idx][p]
            
            # Get colors of other 3
            co = [colors[o] for o in others]
            set_o = set(co)
            size_o = len(set_o)
            
            old_total = size_o + (0 if old_color in set_o else 1)
            new_total = size_o + (0 if new_color in set_o else 1)
            
            old_bad = 1 if old_total <= 2 else 0
            new_bad = 1 if new_total <= 2 else 0
            delta += new_bad - old_bad
        
        return delta
    
    best_solution = None
    best_cost = float('inf')
    
    # Try k = 3, 4, 5, ...
    for k in range(3, 12):
        if time.time() > deadline - 15:
            break
            
        best_k_solution = None
        best_k_cost = float('inf')
        
        # Multiple restarts with different initialization strategies
        num_restarts = 0
        max_restarts = 80
        
        while time.time() < deadline - 10 and num_restarts < max_restarts:
            num_restarts += 1
            restart_start = time.time()
            
            # Different initialization strategies
            if num_restarts % 3 == 1:
                # Random
                colors = np.random.randint(0, k, n)
            elif num_restarts % 3 == 2:
                # Cyclic pattern
                colors = np.array([i % k for i in range(n)], dtype=np.int64)
            else:
                # Block pattern
                colors = np.array([(i // (n // k)) % k for i in range(n)], dtype=np.int64)
            
            current_cost = count_bad(colors)
            
            if current_cost == 0:
                best_k_solution = colors.copy()
                best_k_cost = 0
                break
                
            # Simulated Annealing
            T = 1.0
            T_min = 0.01
            cooling = 0.997
            
            while time.time() - restart_start < 0.7:
                if current_cost == 0:
                    best_k_solution = colors.copy()
                    best_k_cost = 0
                    break
                    
                i = np.random.randint(n)
                old_color = colors[i]
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                    
                delta = compute_delta(colors, i, old_color, new_color)
                
                if delta < 0:
                    colors[i] = new_color
                    current_cost += delta
                else:
                    if random.random() < np.exp(-delta / T):
                        colors[i] = new_color
                        current_cost += delta
                        
                T *= cooling
                if T < T_min:
                    T = T_min

            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()
                
            if best_k_cost == 0:
                break

        if best_k_cost == 0:
            result = list(best_k_solution)
            used_colors = set(result)
            
            # Aggressive color reduction
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                improved = False
                # Try merging each pair of colors
                for c1 in list(used_colors):
                    for c2 in list(used_colors):
                        if c1 == c2:
                            continue
                        # Try mapping c1 -> c2
                        test = [c2 if c == c1 else c for c in result]
                        if count_bad(np.array(test)) == 0:
                            result = test
                            used_colors = set(result)
                            improved = True
                            break
                    if improved:
                        break
                    if time.time() > deadline - 5:
                        break
                if not improved:
                    break
            
            return result
            
    # Fallback: greedy construction
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors
