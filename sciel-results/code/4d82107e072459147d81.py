import time
import random
import numpy as np

def construct():
    N = 128
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill

    # Precompute 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d)
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            # 0-indexed
            ap = (a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d)
            aps.append(ap)
    
    num_aps = len(aps)

    # Precompute APs containing each index for fast delta calculation
    aps_by_idx = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for i in ap:
            aps_by_idx[i].append(idx)

    # Bitmask-based bad count calculation
    # For each AP, we need to check if it uses < 3 distinct colors.
    # Since colors are small integers (0..K-1), we can use bitmasks.
    # However, K can be up to 128, so bitmasks might be large.
    # But we can just count distinct colors in the 4-tuple.
    # A 4-tuple has < 3 distinct colors if:
    # 1. All 4 same (1 distinct)
    # 2. 3 same, 1 diff (2 distinct)
    # 3. 2 same, 2 same (2 distinct)
    
    def is_bad_4ap(c0, c1, c2, c3):
        # Check if number of distinct colors < 3
        # This is equivalent to:
        # (c0==c1 and c2==c3) or 
        # (c0==c2 and c1==c3) or 
        # (c0==c3 and c1==c2) or
        # (c0==c1 and c1==c2) or ... 
        # Simpler: count distinct
        # For speed, manual check:
        # If c0==c1:
        #   if c2==c3: return True (2 distinct or 1)
        #   else: return (c2==c0 or c3==c0)
        # If c0!=c1:
        #   if c0==c2: return (c3==c0 or c3==c1)
        #   if c0==c3: return (c2==c0 or c2==c1)
        #   if c1==c2: return (c3==c1 or c3==c0)
        #   if c1==c3: return (c2==c1 or c2==c0)
        #   if c2==c3: return (c0==c2 or c1==c2)
        #   else: return False
        
        if c0 == c1:
            if c2 == c3:
                return True
            else:
                return (c2 == c0 or c3 == c0)
        else:
            if c0 == c2:
                return (c3 == c0 or c3 == c1)
            if c0 == c3:
                return (c2 == c0 or c2 == c1)
            if c1 == c2:
                return (c3 == c1 or c3 == c0)
            if c1 == c3:
                return (c2 == c1 or c2 == c0)
            if c2 == c3:
                return (c0 == c2 or c1 == c2)
            else:
                return False

    def count_bad_aps(colors):
        bad = 0
        for ap in aps:
            if is_bad_4ap(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]):
                bad += 1
        return bad

    # Greedy initialization
    # We want to minimize the number of colors used.
    # Let's try to construct a valid coloring with a small number of colors.
    # A known result: Van der Waerden number W(3, 4) = 76.
    # This means any 3-coloring of [1, 76] contains a monochromatic 4-AP.
    # But we only need to avoid < 3 distinct colors (i.e., 1 or 2 distinct colors).
    # This is a weaker condition than avoiding monochromatic 4-APs.
    # So 3 colors might be sufficient.
    
    def greedy_init(k):
        colors = [-1] * N
        for i in range(N):
            valid_colors = list(range(k))
            random.shuffle(valid_colors)
            for c in valid_colors:
                ok = True
                # Check if assigning c to i creates a bad 4-AP
                # We only need to check 4-APs where i is the last element?
                # No, i can be any position in the 4-AP.
                # But if we fill left to right, only APs ending at i or involving future elements?
                # Actually, if we fill left to right, any 4-AP involving i and future elements
                # is not yet determined. So we only need to check 4-APs where all other
                # elements are already assigned.
                # That means i must be the largest index in the AP?
                # No, the AP is (a, a+d, a+2d, a+3d). The largest index is a+3d.
                # So if i is not the largest, the AP is not fully assigned yet.
                # Wait, if i is the largest, then a+3d = i, so a = i-3d.
                # We need a >= 0.
                # So for each d, if i-3d >= 0, check the AP (i-3d, i-2d, i-d, i).
                
                for d in range(1, i // 3 + 1):
                    a = i - 3 * d
                    if a < 0:
                        break
                    ap = (a, a + d, a + 2 * d, i)
                    if is_bad_4ap(colors[ap[0]], colors[ap[1]], colors[ap[2]], c):
                        ok = False
                        break
                
                if ok:
                    colors[i] = c
                    break
            if colors[i] == -1:
                # Fallback: assign 0
                colors[i] = 0
        return colors

    best_solution = None
    best_colors_used = N + 1

    # Try k from 3 to 8
    for k in range(3, 9):
        if time.time() - start_time > time_limit - 2:
            break
        
        # Multiple restarts
        max_restarts = 50
        found_valid = False
        
        for r in range(max_restarts):
            if time.time() - start_time > time_limit - 2:
                break
            
            # Initialize
            colors = greedy_init(k)
            
            # Calculate initial bad count
            bad_count = count_bad_aps(colors)
            
            if bad_count == 0:
                num_colors = len(set(colors))
                if num_colors < best_colors_used:
                    best_colors_used = num_colors
                    best_solution = colors[:]
                found_valid = True
                break
            
            # Local Search
            # Objective: minimize bad_count.
            # Move: change color of one index.
            
            temp = 1.0
            cooling = 0.95
            max_iter = 2000
            
            for _ in range(max_iter):
                if time.time() - start_time > time_limit - 2:
                    break
                if bad_count == 0:
                    break
                
                # Pick random index
                i = random.randint(0, N - 1)
                old_c = colors[i]
                
                # Pick new color
                best_new_c = old_c
                best_new_bad = bad_count
                best_delta = 0
                
                # Try all k colors
                for new_c in range(k):
                    if new_c == old_c:
                        continue
                    
                    # Calculate delta in bad_count
                    delta = 0
                    for ap_idx in aps_by_idx[i]:
                        ap = aps[ap_idx]
                        c0 = colors[ap[0]]
                        c1 = colors[ap[1]]
                        c2 = colors[ap[2]]
                        c3 = colors[ap[3]]
                        
                        was_bad = is_bad_4ap(c0, c1, c2, c3)
                        
                        nc0 = new_c if ap[0] == i else c0
                        nc1 = new_c if ap[1] == i else c1
                        nc2 = new_c if ap[2] == i else c2
                        nc3 = new_c if ap[3] == i else c3
                        
                        is_bad_now = is_bad_4ap(nc0, nc1, nc2, nc3)
                        
                        if was_bad and not is_bad_now:
                            delta -= 1
                        elif not was_bad and is_bad_now:
                            delta += 1
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_c = new_c
                        best_new_bad = bad_count + delta
                
                # Accept best move
                if best_new_c != old_c:
                    colors[i] = best_new_c
                    bad_count = best_new_bad
                else:
                    # No improvement, maybe accept worse move with prob exp(-delta/temp)?
                    # Or just restart if stuck.
                    pass
                
                temp *= cooling
            
            if bad_count == 0:
                num_colors = len(set(colors))
                if num_colors < best_colors_used:
                    best_colors_used = num_colors
                    best_solution = colors[:]
                found_valid = True
                break
        
        if found_valid:
            # If we found a valid solution with k colors, we can stop
            # because we are trying to minimize colors.
            # But wait, maybe a solution with k+1 colors uses fewer distinct colors?
            # No, if we use k colors, the number of distinct colors is at most k.
            # If we find a valid solution with k=3, the min distinct colors is 3.
            # If k=3 fails, we try k=4. If k=4 works, min distinct colors is <=4.
            # So we should stop if we find a valid solution with the smallest k.
            break

    if best_solution:
        return best_solution
    else:
        # Fallback: Return a trivial valid coloring (all distinct)
        colors = list(range(N))
        return colors
