import random
import time
import math

def construct():
    n = 256
    
    # Precompute all 4-APs
    all_aps = []
    pos_to_aps = [[] for _ in range(n)]
    # For greedy construction: aps where i is the last element (index 3)
    aps_ending_at = [[] for _ in range(n)]
    
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(0, n - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(all_aps)
            all_aps.append(ap)
            for j in range(4):
                pos = ap[j]
                pos_to_aps[pos].append((idx, j))
            # The last element is ap[3]
            aps_ending_at[ap[3]].append(idx)
            
    # Helper to count current mono APs
    def count_mono(colors):
        count = 0
        for ap in all_aps:
            if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count

    # Helper to calculate delta if we flip index i
    def calc_delta(colors, i):
        old_c = colors[i]
        new_c = 1 - old_c
        delta = 0
        for ap_idx, pos_in_ap in pos_to_aps[i]:
            ap = all_aps[ap_idx]
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            
            before_mono = (c0 == c1 == c2 == c3)
            
            if pos_in_ap == 0:
                after_mono = (new_c == c1 == c2 == c3)
            elif pos_in_ap == 1:
                after_mono = (c0 == new_c == c2 == c3)
            elif pos_in_ap == 2:
                after_mono = (c0 == c1 == new_c == c3)
            else:
                after_mono = (c0 == c1 == c2 == new_c)
                
            if before_mono and not after_mono:
                delta -= 1
            elif not before_mono and after_mono:
                delta += 1
        return delta

    # Phase 1: Greedy Construction
    colors = [-1] * n
    for i in range(n):
        cost_0 = 0
        cost_1 = 0
        for ap_idx in aps_ending_at[i]:
            ap = all_aps[ap_idx]
            # The other 3 elements
            p0, p1, p2 = ap[0], ap[1], ap[2]
            # They must be colored already because p0,p1,p2 < i
            c0, c1, c2 = colors[p0], colors[p1], colors[p2]
            
            # If any are -1, this AP is not fully determined yet?
            # Wait, aps_ending_at[i] contains APs where i is the LAST element.
            # So p0, p1, p2 are always < i.
            # So they are always colored in the greedy phase.
            
            if c0 == c1 == c2 == 0:
                cost_0 += 1
            elif c0 == c1 == c2 == 1:
                cost_1 += 1
            # If they are mixed, flipping i to 0 or 1 won't create a mono AP
            # because the other 3 are not all 0 or all 1.
            
        if cost_0 < cost_1:
            colors[i] = 0
        elif cost_1 < cost_0:
            colors[i] = 1
        else:
            colors[i] = random.randint(0, 1)
            
    current_cost = count_mono(colors)
    best_cost = current_cost
    best_colors = colors.copy()
    
    start_time = time.time()
    time_limit = 115.0
    
    # Phase 2: Simulated Annealing Refinement
    T = 5.0
    cooling = 0.999
    min_T = 0.01
    
    while T > min_T and (time.time() - start_time) < time_limit:
        i = random.randint(0, n - 1)
        d_cost = calc_delta(colors, i)
        
        if d_cost < 0 or random.random() < math.exp(-d_cost / T):
            colors[i] = 1 - colors[i]
            current_cost += d_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                
        T *= cooling
        
    return best_colors
