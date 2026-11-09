import random
import time
import math

def construct():
    n = 256
    
    # Precompute all 4-APs
    # AP is (a, a+d, a+2d, a+3d)
    all_aps = []
    # For efficient delta calculation, we need to know which APs each index is part of
    # pos_to_aps[i] = list of (ap_index, position_in_ap)
    pos_to_aps = [[] for _ in range(n)]
    
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(0, n - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(all_aps)
            all_aps.append(ap)
            for j in range(4):
                pos = ap[j]
                pos_to_aps[pos].append((idx, j))
                
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
            # Check if AP is mono before flip
            # We can just check all 4 colors
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            
            # Before
            before_mono = (c0 == c1 == c2 == c3)
            
            # After
            # Only one element changes
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

    # Simulated Annealing
    # Initial state: Random
    colors = [random.randint(0, 1) for _ in range(n)]
    current_cost = count_mono(colors)
    best_cost = current_cost
    best_colors = colors.copy()
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    while T > min_T and (time.time() - start_time) < time_limit:
        # Pick random index
        i = random.randint(0, n - 1)
        
        # Calculate delta
        d_cost = calc_delta(colors, i)
        
        # Acceptance criterion
        if d_cost < 0 or random.random() < math.exp(-d_cost / T):
            colors[i] = 1 - colors[i]
            current_cost += d_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                
        T *= cooling
        
    return best_colors
