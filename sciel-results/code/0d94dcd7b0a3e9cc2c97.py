import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    n = 128
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    # Map index to list of AP indices containing it
    # Also precompute for each (ap_idx, pos_in_ap) the other 3 positions to avoid .index() call
    idx_to_aps = [[] for _ in range(n)]
    # other_indices[ap_idx][pos_idx] = tuple of other 3 indices
    other_indices = []
    for i, ap in enumerate(aps):
        for pos_idx, idx in enumerate(ap):
            idx_to_aps[idx].append(i)
        
        # Precompute others for this AP
        ap_others = []
        for pos_idx in range(4):
            others = tuple(ap[j] for j in range(4) if j != pos_idx)
            ap_others.append(others)
        other_indices.append(ap_others)

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
    # The seed program uses a greedy fallback that likely uses more colors.
    # We want to minimize k.
    # Let's try k=3 first.
    
    for k in range(3, 15):
        if time.time() > deadline - 15:
            break
            
        best_k_solution = None
        best_k_cost = float('inf')
        
        # Multiple restarts
        # Each restart should run for a fixed time or until convergence
        restart_time_limit = 1.5  # seconds per restart
        
        while time.time() < deadline - 10:
            restart_start = time.time()
            
            # Random initialization
            colors = np.random.randint(0, k, n)
            current_cost = count_bad_colors(colors)
            
            if current_cost == 0:
                best_k_solution = colors.copy()
                best_k_cost = 0
                break
                
            # Simulated Annealing
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
                old_color = colors[i]
                new_color = np.random.randint(0, k)
                if new_color == old_color:
                    continue
                    
                # Calculate delta cost
                delta = 0
                for ap_idx in idx_to_aps[i]:
                    ap = aps[ap_idx]
                    # Find position of i in ap
                    # We can precompute this mapping: pos_in_ap[ap_idx][i]
                    # But i is not in ap_idx directly in a 2D array easily.
                    # Let's use the precomputed other_indices.
                    # We need to know which position in the AP is i.
                    # Since we stored idx_to_aps, we know i is in ap_idx.
                    # We can check which of the 4 positions is i.
                    # To avoid 4 comparisons, we can precompute pos_in_ap[ap_idx][i]
                    # But i varies.
                    # Let's just do the 4 comparisons, it's fast.
                    pos_in_ap = -1
                    if ap[0] == i: pos_in_ap = 0
                    elif ap[1] == i: pos_in_ap = 1
                    elif ap[2] == i: pos_in_ap = 2
                    else: pos_in_ap = 3
                    
                    others = other_indices[ap_idx][pos_in_ap]
                    
                    # Get colors of others
                    c_others = [colors[j] for j in others]
                    
                    # Old set: c_others + [old_color]
                    # New set: c_others + [new_color]
                    
                    # Optimization: 
                    # If old_color is in c_others, removing it from set might reduce size, or not.
                    # If new_color is in c_others, adding it might not increase size.
                    
                    # Let's just compute sets. 
                    # To be faster:
                    # distinct_old = len(set(c_others + [old_color]))
                    # distinct_new = len(set(c_others + [new_color]))
                    
                    # We can check if distinct <= 2.
                    # distinct_old <= 2 iff all elements in c_others are in {old_color, x} for some x?
                    # Simpler: 
                    # s_old = set(c_others)
                    # s_old.add(old_color)
                    # s_new = set(c_others)
                    # s_new.add(new_color)
                    
                    # Even faster:
                    # If old_color == new_color, delta is 0. (Handled)
                    # If old_color is not in c_others:
                    #   s_old_size = len(set(c_others)) + 1
                    # Else:
                    #   s_old_size = len(set(c_others))
                    # Same for new.
                    
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
                        # If stuck for a long time, we could break, but we have time limit.
                        # Let's just continue to use the time.
                        
                # Cool down
                T *= 0.999
                if T < 0.01:
                    T = 0.01

            if current_cost < best_k_cost:
                best_k_cost = current_cost
                best_k_solution = colors.copy()
                
            if best_k_cost == 0:
                break

        if best_k_cost == 0:
            # Found a valid solution with k colors
            return list(best_k_solution)
            
    # Fallback: Greedy construction similar to seed
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors
