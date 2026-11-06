"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n = 128
    start_time = time.time()
    
    # Precompute 4-term APs
    aps = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            if a + 3 * d < n:
                aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each position, list of AP indices that include it
    pos_aps = [[] for _ in range(n)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_aps[a].append(idx)
        pos_aps[b].append(idx)
        pos_aps[c].append(idx)
        pos_aps[d].append(idx)
    
    # Try minimum colors
    for num_colors in range(3, 128):
        if time.time() - start_time > 55:
            break
        colors = [-1] * n
        
        def is_valid_assignment(pos, color):
            for ap_idx in pos_aps[pos]:
                a, b, c, d = aps[ap_idx]
                vals = []
                for p in (a, b, c, d):
                    if p == pos:
                        vals.append(color)
                    else:
                        vals.append(colors[p])
                if -1 not in vals:
                    if len(set(vals)) < 3:
                        return False
            return True
        
        # Order positions by constraint count
        order = sorted(range(n), key=lambda i: -len(pos_aps[i]))
        
        def backtrack(k):
            if time.time() - start_time > 55:
                return False
            if k == n:
                return True
            pos = order[k]
            for color in range(num_colors):
                if is_valid_assignment(pos, color):
                    colors[pos] = color
                    if backtrack(k + 1):
                        return True
                    colors[pos] = -1
            return False
        
        if backtrack(0):
            return [int(c) for c in colors]
    
    # Fallback
    return list(range(n))
# EVOLVE-BLOCK-END
