import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s margin out of 90s wall-clock
    
    n = 128
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    # Map index to list of AP indices containing it
    idx_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for idx in ap:
            idx_to_aps[idx].append(i)
    
    # Precompute pos_in_ap mapping: pos_in_ap[ap_idx][idx] = position of idx in ap
    pos_in_ap = np.full((num_aps, n), -1, dtype=np.int8)
    for i, ap in enumerate(aps):
        for pos_idx, idx in enumerate(ap):
            pos_in_ap[i][idx] = pos_idx

    def count_bad_colors(colors):
        """Count APs with <= 2 distinct colors using vectorized numpy."""
        c = colors[aps_arr]  # shape (num_aps, 4)
        not_012 = ~((c[:,0]!=c[:,1]) & (c[:,1]!=c[:,2]) & (c[:,0]!=c[:,2]))
        not_013 = ~((c[:,0]!=c[:,1]) & (c[:,1]!=c[:,3]) & (c[:,0]!=c[:,3]))
        not_023 = ~((c[:,0]!=c[:,2]) & (c[:,2]!=c[:,3]) & (c[:,0]!=c[:,3]))
        not_123 = ~((c[:,1]!=c[:,2]) & (c[:,2]!=c[:,3]) & (c[:,1]!=c[:,3]))
        bad_mask = not_012 & not_013 & not_023 & not_123
        return int(np.sum(bad_mask))

    best_solution = None
    best_cost = float('inf')
    
    # Try k = 3, 4, 5...
    for k in range(3, 15):
        if time.time() > deadline - 10:
            break
            
        best_k_solution = None
        best_k_cost = float('inf')
        
        # Multiple restarts
        restart_time_limit = 1.2  # seconds per restart
        
        while time.time() < deadline - 5:
            restart_start = time.time()
            
            # Random initialization
            colors = np.random.randint(0, k, n).astype(np.int32)
            current_cost = count_bad_colors(colors)
            
            if current_cost == 0:
                best_k_solution = colors.copy()
                best_k_cost = 0
                break
                
            # Simulated Annealing with faster cooling
            T = 1.0
            no_improve = 0
            # Run until time limit or convergence
            while time.time() - restart_start < restart_time_limit:
                if current_cost == 0:
                    best_k_solution = colors.copy()
                    best_k_cost = 0
                    break
                    
                # Pick random index
                i = np.random.randint(n)
                old_color = int(colors[i])
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                    
                # Calculate delta cost using precomputed pos_in_ap
                delta = 0
                for ap_idx in idx_to_aps[i]:
                    pos = pos_in_ap[ap_idx][i]
                    # Get the other 3 positions in this AP
                    ap = aps[ap_idx]
                    others = tuple(ap[j] for j in range(4) if j != pos)
                    
                    # Get colors of others
                    c_others = [int(colors[j]) for j in others]
                    
                    set_others = set(c_others)
                    size_others = len(set_others)
                    
                    old_size = size_others + (0 if old_color in set_others else 1)
                    new_size = size_others + (0 if new_color in set_others else 1)
                    
                    old_bad = 1 if old_size <= 2 else 0
                    new_bad = 1 if new_size <= 2 else 0
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
                        
                # Faster cool down
                T *= 0.995
                if T < 0.005:
                    T = 0.005

            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()
                
            if best_k_cost == 0:
                break

        if best_k_cost == 0:
            # Found a valid solution with k colors
            # Try to reduce the number of distinct colors used
            result = list(best_k_solution)
            used_colors = set(result)
            
            # Try to map each color to a smaller set
            for _ in range(200):  # 200 attempts to reduce colors
                if time.time() > deadline - 2:
                    break
                # Try to find a color that can be remapped
                for color in list(used_colors):
                    # Try to remap color to another existing color
                    for target_color in used_colors:
                        if color == target_color:
                            continue
                        # Check if remapping is valid
                        test_colors = np.array([target_color if c == color else c for c in result], dtype=np.int32)
                        if count_bad_colors(test_colors) == 0:
                            result = list(test_colors)
                            used_colors = set(result)
                            break
                    if time.time() > deadline - 2:
                        break
            
            return result
            
    # Fallback: Greedy construction similar to seed
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors
