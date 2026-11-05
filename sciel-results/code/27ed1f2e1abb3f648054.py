"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

def construct():
    n = 128
    # Precompute 4-APs grouped by their maximum element
    # aps_by_max[max_val] = list of tuples (a, b, c) such that a, b, c, max_val is a 4-AP
    aps_by_max = [[] for _ in range(n + 1)]
    for a in range(1, n + 1):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                aps_by_max[e].append((a, b, c))
    
    colors = [-1] * (n + 1) # 1-indexed
    
    # Optimization: Order of checking colors? 
    # We want to minimize the number of colors used.
    # Trying 0, 1, 2... ensures we use the smallest possible labels.
    
    def backtrack(x):
        if x > n:
            return True
        
        for c in range(128):
            valid = True
            for (a, b, d_pos) in aps_by_max[x]:
                ca = colors[a]
                cb = colors[b]
                cd = colors[d_pos]
                
                # Check if adding c to {ca, cb, cd} results in < 3 distinct colors
                # If |{ca, cb, cd}| >= 3, it's always valid.
                # If |{ca, cb, cd}| == 2, c must not be in {ca, cb, cd}.
                # If |{ca, cb, cd}| == 1, c must not be in {ca, cb, cd}.
                
                # Quick check
                if ca == cb and cb == cd:
                    if c == ca:
                        valid = False
                        break
                elif ca == cb or ca == cd or cb == cd:
                    # Two are same, one different, or all same handled above
                    # Set size is 2
                    # c cannot be any of the existing colors
                    if c == ca or c == cb or c == cd:
                        valid = False
                        break
                else:
                    # All 3 distinct, size 3
                    # Always valid
                    pass
            
            if valid:
                colors[x] = c
                if backtrack(x + 1):
                    return True
                colors[x] = -1
        
        return False
    
    if backtrack(1):
        return [colors[i] for i in range(1, n + 1)]
    else:
        # Fallback: return a valid coloring if found by other means, or just return the array
        # This should not happen for n=128 with 128 colors available
        return [0] * n

# EVOLVE-BLOCK-END
