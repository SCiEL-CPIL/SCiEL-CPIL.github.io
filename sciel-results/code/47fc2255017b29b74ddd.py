import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 128
    # Precompute APs
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    num_aps = len(aps)
    
    # Precompute structures for fast delta calculation
    idx_to_aps = [[] for _ in range(n)]
    # other_indices[ap_idx][pos_in_ap] = tuple of other 3 indices
    other_indices = []
    
    for i, ap in enumerate(aps):
        for pos_idx, idx in enumerate(ap):
            idx_to_aps[idx].append(i)
            
        ap_others = []
        for pos_idx in range(4):
            others = tuple(ap[j] for j in range(4) if j != pos_idx)
            ap_others.append(others)
        other_indices.append(ap_others)
        
    # pos_in_ap[ap_idx][idx] = position of idx in ap
    pos_in_ap = [[-1]*n for _ in range(num_aps)]
    for i, ap in enumerate(aps):
        for pos_idx, idx in enumerate(ap):
            pos_in_ap[i][idx] = pos_idx

    def count_bad_colors(colors):
        """Count APs with <= 2 distinct colors."""
        bad = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            # Fast check for <= 2 distinct colors
            # If all same: bad
            # If 2 distinct: bad
            # We can just use set
            if len({c0, c1, c2, c3}) <= 2:
                bad += 1
        return bad

    def is_bad_ap(c0, c1, c2, c3):
        # Return 1 if bad, 0 if good
        # Check if set size <= 2
        # Optimized check:
        if c0 == c1:
            if c2 == c3:
                return 1
            if c2 == c0 or c3 == c0:
                return 1
            # c2 != c0, c3 != c0, c2 != c3
            # Colors: c0, c0, c2, c3. Distinct: {c0, c2, c3}. Size 3. Good.
            return 0
        elif c0 == c2:
            if c1 == c3:
                return 1
            if c1 == c0 or c3 == c0:
                return 1
            return 0
        elif c0 == c3:
            if c1 == c2:
                return 1
            if c1 == c0 or c2 == c0:
                return 1
            return 0
        else: # c0 != c1, c0 != c2, c0 != c3
            if c1 == c2:
                if c3 == c1:
                    return 1
                # c1==c2, c3!=c1, c0!=c1. Distinct: {c0, c1, c3}. Size 3. Good.
                return 0
            elif c1 == c3:
                if c2 == c1:
                    return 1
                return 0
            elif c2 == c3:
                if c1 == c2:
                    return 1
                return 0
            else:
                # All different? Or c1!=c2!=c3!=c1 and none equal c0.
                # Distinct: {c0, c1, c2, c3}. Size 4. Good.
                return 0

    # Local Search
    best_solution = None
    best_cost = float('inf')
    
    # Try k=3 first
    for k in [3, 4, 5]:
        time_limit = 20 if k == 3 else 15
        if time.time() > deadline - 5:
            break
            
        best_k_solution = None
        best_k_cost = float('inf')
        
        restarts = 0
        while time.time() < deadline - 5 - (10 if k > 3 else 0):
            restarts += 1
            colors = np.random.randint(0, k, n)
            current_cost = count_bad_colors(colors)
            
            if current_cost == 0:
                best_k_solution = colors.copy()
                best_k_cost = 0
                break
                
            T = 1.0
            min_T = 0.01
            cooling = 0.999
            no_improve = 0
            max_no_improve = 500
            
            # Inner loop
            while time.time() < deadline - 5 - (10 if k > 3 else 0):
                if current_cost == 0:
                    best_k_solution = colors.copy()
                    best_k_cost = 0
                    break
                    
                i = np.random.randint(n)
                old_color = colors[i]
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                    
                delta = 0
                for ap_idx in idx_to_aps[i]:
                    pos = pos_in_ap[ap_idx][i]
                    others = other_indices[ap_idx][pos]
                    c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
                    
                    old_bad = is_bad_ap(c1, c2, c3, old_color)
                    new_bad = is_bad_ap(c1, c2, c3, new_color)
                    delta += new_bad - old_bad
                    
                if delta < 0:
                    colors[i] = new_color
                    current_cost += delta
                    no_improve = 0
                else:
                    if random.random() < np.exp(-delta / T):
                        colors[i] = new_color
                        current_cost += delta
                    else:
                        no_improve += 1
                        if no_improve > max_no_improve:
                            break
                            
                T *= cooling
                if T < min_T:
                    T = min_T

            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()
                
            if best_k_cost == 0:
                break

        if best_k_cost == 0:
            return list(best_k_solution)

    # Fallback
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors
