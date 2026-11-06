import time
import random
import numpy as np

def construct():
    N = 128
    start_time = time.time()
    time_limit = 50.0  # Safety margin before 60s kill

    # Precompute 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d)
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            # 0-indexed
            ap = (a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d)
            aps.append(ap)
            
    # Precompute APs containing each index for fast delta calculation
    aps_by_idx = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for i in ap:
            aps_by_idx[i].append(idx)

    # Helper to count distinct colors in a 4-AP given colors array
    # We can use a bitmask trick if colors are small, but simple set is okay for N=128 if optimized.
    # Better: Precompute nothing, just check on the fly.
    
    def count_bad_aps(colors):
        bad = 0
        for ap in aps:
            c = [colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
            if len(set(c)) < 3:
                bad += 1
        return bad

    def get_bad_aps_list(colors):
        bad_list = []
        for idx, ap in enumerate(aps):
            c = [colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]]
            if len(set(c)) < 3:
                bad_list.append(idx)
        return bad_list

    # Greedy initialization avoiding monochromatic 3-APs
    # This is a stronger condition than avoiding mono-4-APs, but helps.
    # We use k colors.
    def greedy_init(k):
        colors = [-1] * N
        # For each position, check if assigning color c creates a mono-3-AP
        # A mono-3-AP ending at i: (i-2d, i-d, i)
        # A mono-3-AP starting at i: (i, i+d, i+2d)
        # A mono-3-AP middle at i: (i-d, i, i+d)
        
        # To be safe and fast, we just check all 3-APs involving i.
        # Precompute 3-APs?
        # Or just check locally.
        
        # Let's use a simpler greedy: assign random valid color.
        # Valid: does not form a mono-3-AP with any existing neighbor.
        
        # Precompute 3-AP constraints for speed
        # For each i, list of (j, k) such that (i, j, k) is a 3-AP.
        # Actually, just checking neighbors is O(N^2) worst case, which is fine for N=128.
        
        for i in range(N):
            valid_colors = list(range(k))
            random.shuffle(valid_colors)
            for c in valid_colors:
                ok = True
                # Check 3-APs where i is the last term: (i-2d, i-d, i)
                for d in range(1, i // 2 + 1):
                    j = i - d
                    k_idx = i - 2 * d
                    if colors[j] == c and colors[k_idx] == c:
                        ok = False
                        break
                if not ok:
                    continue
                # Check 3-APs where i is the first term: (i, i+d, i+2d)
                # These involve future unassigned nodes, so no conflict yet.
                # Check 3-APs where i is the middle term: (i-d, i, i+d)
                for d in range(1, N - i):
                    j = i - d
                    k_idx = i + d
                    if j >= 0 and k_idx < N:
                        if colors[j] == c and colors[k_idx] == c:
                            ok = False
                            break
                if not ok:
                    continue
                
                # Also check if it creates a bad 4-AP? 
                # Greedy init doesn't guarantee 4-AP validity, just 3-AP mono validity.
                # We will fix 4-APs in local search.
                colors[i] = c
                break
            if colors[i] == -1:
                # Fallback: assign 0 (should rarely happen with k>=3)
                colors[i] = 0
        return colors

    best_solution = None
    best_colors_used = N + 1

    # Try k from 3 to 6
    for k in range(3, 7):
        if time.time() - start_time > time_limit - 2:
            break
            
        # Multiple restarts
        t_start_k = time.time()
        max_restarts = 20
        
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
                break # Found valid for this k, move to next k? 
                      # Actually, we want to minimize colors. If k=3 works, that's best.
                      # But maybe we can find a solution with fewer *distinct* colors 
                      # if we used k=4 but only used 3? 
                      # No, if k=3 works, min colors is 3.
                      # If k=3 fails, we try k=4. If k=4 works, min colors is <=4.
                      # We should stop if we find a valid solution with k=3.
                      # If k=3 fails, we keep trying k=4.
            
            # Local Search
            # Objective: minimize bad_count.
            # Move: change color of one index.
            
            temp = 1.0
            cooling = 0.95
            t = time.time()
            max_iter = 5000
            
            for _ in range(max_iter):
                if time.time() - start_time > time_limit - 2:
                    break
                if bad_count == 0:
                    break
                    
                # Pick random index
                i = random.randint(0, N - 1)
                old_c = colors[i]
                
                # Pick new color
                # We want to stay within k colors, or maybe expand? 
                # To minimize distinct colors, we should stick to the current set of colors used.
                # Let's get the set of currently used colors.
                used_colors = set(colors)
                # If we want to minimize, we shouldn't introduce new colors unless necessary.
                # But local search usually fixes bad APs by changing to an existing color.
                # Let's try all colors in range(k).
                
                best_new_c = old_c
                best_new_bad = bad_count
                best_delta = 0
                
                # Optimization: Only check colors that might help.
                # Just iterate all k colors.
                for new_c in range(k):
                    if new_c == old_c:
                        continue
                    
                    # Calculate delta in bad_count
                    delta = 0
                    for ap_idx in aps_by_idx[i]:
                        ap = aps[ap_idx]
                        # Check old state
                        c0 = colors[ap[0]]
                        c1 = colors[ap[1]]
                        c2 = colors[ap[2]]
                        c3 = colors[ap[3]]
                        
                        # Quick check: was it bad?
                        # We need to know if it was bad before and after.
                        # To avoid set creation, we can use a helper or inline logic.
                        # Inline logic for "len(set) < 3":
                        # Bad if:
                        # 1. All 4 same
                        # 2. 3 same, 1 diff (e.g., A A A B)
                        # 3. 2 same, 2 same (e.g., A A B B)
                        
                        # Let's just use a small helper for speed in Python? 
                        # No, inline is faster if we avoid function call overhead.
                        
                        # Old bad check
                        if c0 == c1 == c2 == c3:
                            was_bad = True
                        elif c0 == c1 == c2 and c3 != c0:
                            was_bad = True
                        elif c0 == c1 == c3 and c2 != c0:
                            was_bad = True
                        elif c0 == c2 == c3 and c1 != c0:
                            was_bad = True
                        elif c1 == c2 == c3 and c0 != c1:
                            was_bad = True
                        elif c0 == c1 and c2 == c3 and c0 != c2:
                            was_bad = True
                        elif c0 == c2 and c1 == c3 and c0 != c1:
                            was_bad = True
                        elif c0 == c3 and c1 == c2 and c0 != c1:
                            was_bad = True
                        else:
                            was_bad = False

                        # New state
                        nc0 = new_c if ap[0] == i else c0
                        nc1 = new_c if ap[1] == i else c1
                        nc2 = new_c if ap[2] == i else c2
                        nc3 = new_c if ap[3] == i else c3
                        
                        if nc0 == nc1 == nc2 == nc3:
                            is_bad_now = True
                        elif nc0 == nc1 == nc2 and nc3 != nc0:
                            is_bad_now = True
                        elif nc0 == nc1 == nc3 and nc2 != nc0:
                            is_bad_now = True
                        elif nc0 == nc2 == nc3 and nc1 != nc0:
                            is_bad_now = True
                        elif nc1 == nc2 == nc3 and nc0 != nc1:
                            is_bad_now = True
                        elif nc0 == nc1 and nc2 == nc3 and nc0 != nc2:
                            is_bad_now = True
                        elif nc0 == nc2 and nc1 == nc3 and nc0 != nc1:
                            is_bad_now = True
                        elif nc0 == nc3 and nc1 == nc2 and nc0 != nc1:
                            is_bad_now = True
                        else:
                            is_bad_now = False

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
                break

    if best_solution:
        return best_solution
    else:
        # Fallback: Return a random coloring or greedy one
        # This should not happen if k is large enough (e.g., k=128 trivially works)
        # But we want to minimize.
        # If no valid solution found within time, return the best partial? 
        # The task says "invalid result... scores 0".
        # So we MUST return a valid coloring.
        # A trivial valid coloring uses 128 colors (all distinct).
        # Let's construct a trivial one if search fails.
        colors = list(range(N))
        return colors
