import time
import random

def construct():
    start_time = time.time()
    # Hard deadline to ensure we return something
    hard_deadline = start_time + 55
    
    N = 128
    colors = [-1] * (N + 1)
    
    # Precompute all 4-APs ending at each x
    # x = a + 3d => a = x - 3d
    aps_ending_at = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for d in range(1, (x - 1) // 3 + 1):
            a = x - 3 * d
            b = a + d
            c = a + 2 * d
            aps_ending_at[x].append((a, b, c))
            
    def check_validity(x, val):
        for a, b, c in aps_ending_at[x]:
            s = {colors[a], colors[b], colors[c], val}
            if len(s) < 3:
                return False
        return True

    def backtrack(x):
        if time.time() > hard_deadline - 5: # 5s margin for cleanup
            return False
            
        if x > N:
            return True
            
        # Determine possible colors and forced color
        possible_colors = {0, 1, 2}
        forced_color = -1
        
        for a, b, c in aps_ending_at[x]:
            ca = colors[a]
            cb = colors[b]
            cc = colors[c]
            s_prev = {ca, cb, cc}
            
            if len(s_prev) == 1:
                # The 4th color cannot be the same as the 3
                possible_colors.discard(ca)
            elif len(s_prev) == 2:
                # The 4th color MUST be the one not in s_prev
                # s_prev has 2 elements from {0,1,2}
                missing = {0, 1, 2} - s_prev
                if len(missing) == 1:
                    m = list(missing)[0]
                    if forced_color == -1:
                        forced_color = m
                    elif forced_color != m:
                        return False
                    # Restrict possible_colors to just m
                    possible_colors = {m}
                else:
                    # Should not happen with 3 colors
                    return False
            else: # len(s_prev) == 3
                # Any color is allowed regarding this AP's color count constraint
                # (Set will be size 3 or 4)
                pass
        
        # If a color was forced, try only that
        if forced_color != -1:
            if check_validity(x, forced_color):
                colors[x] = forced_color
                if backtrack(x + 1):
                    return True
                colors[x] = -1
                return False
            else:
                return False

        # Try other possible colors
        # Order: 0, 1, 2
        for c_val in [0, 1, 2]:
            if c_val not in possible_colors:
                continue
            if check_validity(x, c_val):
                colors[x] = c_val
                if backtrack(x + 1):
                    return True
                colors[x] = -1
                if time.time() > hard_deadline - 5:
                    return False
        return False

    # Attempt 3-coloring
    success = backtrack(1)
    
    if success:
        return [colors[i] for i in range(1, N + 1)]
    
    # Fallback: Greedy 4-coloring
    # This is very fast and should always find a solution
    colors = [-1] * (N + 1)
    for x in range(1, N + 1):
        found = False
        for c_val in [0, 1, 2, 3]:
            if check_validity(x, c_val):
                colors[x] = c_val
                found = True
                break
        if not found:
            # Should not happen, but just in case
            colors[x] = 0
    return [colors[i] for i in range(1, N + 1)]

