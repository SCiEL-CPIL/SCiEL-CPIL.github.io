import time
import random
import math

def construct():
    n = 256
    # Deadline: 110 seconds to be safe
    deadline = time.time() + 110
    
    # 1. Precompute all 4-APs
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    # Inverse index
    elem_aps = [[] for _ in range(n)]
    for i, ap in enumerate(ap_list):
        for elem in ap:
            elem_aps[elem].append(i)
            
    # Helper to initialize state
    def init_state(colors):
        ap_counts = [[0, 0] for _ in range(num_aps)]
        ap_mono = [0] * num_aps
        num_mono = 0
        for i in range(num_aps):
            ap = ap_list[i]
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            cnt0 = (c0 == 0) + (c1 == 0) + (c2 == 0) + (c3 == 0)
            cnt1 = 4 - cnt0
            ap_counts[i][0] = cnt0
            ap_counts[i][1] = cnt1
            if cnt0 == 4 or cnt1 == 4:
                ap_mono[i] = 1
                num_mono += 1
            else:
                ap_mono[i] = 0
        return ap_counts, ap_mono, num_mono

    best_score = float('inf')
    best_colors = None
    
    # Strategy: 3 restarts
    # Each restart gets ~30 seconds
    # Actually, let's do dynamic time allocation.
    # First restart: 40s, second: 40s, third: 20s?
    # Or just loop until deadline.
    
    restarts = 0
    max_restarts = 3
    
    while time.time() < deadline and restarts < max_restarts:
        restarts += 1
        # Random init
        colors = [random.randint(0, 1) for _ in range(n)]
        ap_counts, ap_mono, num_mono = init_state(colors)
        
        current_best = num_mono
        current_best_colors = colors[:]
        
        T = 0.5
        cooling = 0.99995
        min_T = 0.05
        
        # Local search
        while time.time() < deadline - 5: # Stop 5s before end to handle last restart or exit
            idx = random.randrange(n)
            old_color = colors[idx]
            new_color = 1 - old_color
            affected = elem_aps[idx]
            
            delta = 0
            # Calculate delta
            for ap_idx in affected:
                c0, c1 = ap_counts[ap_idx]
                was_mono = ap_mono[ap_idx]
                
                if old_color == 0:
                    new_c0 = c0 - 1
                    new_c1 = c1 + 1
                else:
                    new_c0 = c0 + 1
                    new_c1 = c1 - 1
                
                will_be_mono = 1 if (new_c0 == 4 or new_c1 == 4) else 0
                
                if was_mono and not will_be_mono:
                    delta -= 1
                elif not was_mono and will_be_mono:
                    delta += 1
            
            # Accept
            if delta < 0:
                accept = True
            else:
                if T < 0.01:
                    accept = False
                else:
                    ratio = delta / T
                    if ratio > 20:
                        accept = False
                    else:
                        accept = random.random() < math.exp(-ratio)
            
            if accept:
                colors[idx] = new_color
                
                # Update state
                for ap_idx in affected:
                    c0, c1 = ap_counts[ap_idx]
                    if old_color == 0:
                        c0 -= 1
                        c1 += 1
                    else:
                        c0 += 1
                        c1 -= 1
                    ap_counts[ap_idx] = [c0, c1]
                    
                    was_mono = ap_mono[ap_idx]
                    is_mono = 1 if (c0 == 4 or c1 == 4) else 0
                    ap_mono[ap_idx] = is_mono
                    
                    if was_mono and not is_mono:
                        num_mono -= 1
                    elif not was_mono and is_mono:
                        num_mono += 1
                
                if num_mono < current_best:
                    current_best = num_mono
                    current_best_colors = colors[:]
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        if current_best < best_score:
            best_score = current_best
            best_colors = current_best_colors[:]
            
    if best_colors is None:
        # Fallback
        return [0] * n
        
    return best_colors
