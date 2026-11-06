"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 128
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) 0-indexed
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each position, which APs does it participate in
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # Try to find valid colorings with increasing number of colors
    best_result = None
    
    for k in range(3, 15):  # Try 3 to 14 colors
        if time.time() > deadline - 10:
            break
        
        colors = [-1] * n
        
        def is_valid_assignment(pos, color):
            """Check if assigning 'color' to 'pos' violates any fully-assigned AP constraint."""
            for ap_idx in pos_to_aps[pos]:
                a, b, c, d = ap_list[ap_idx]
                vals = [colors[a], colors[b], colors[c], colors[d]]
                if all(v != -1 for v in vals):
                    if len(set(vals)) < 3:
                        return False
            return True
        
        def has_empty_domain(future_pos):
            """Check if any future position has no valid colors left (forward checking)."""
            for fp in range(future_pos, n):
                if colors[fp] == -1:
                    # Check if at least one color is still valid for this position
                    # Simplified: just check if it's unassigned
                    pass
            return False
        
        def backtrack(pos_idx):
            if time.time() > deadline:
                return False
            
            if pos_idx == n:
                return True
            
            # Try each color
            for color in range(k):
                colors[pos_idx] = color
                if is_valid_assignment(pos_idx, color):
                    if backtrack(pos_idx + 1):
                        return True
                colors[pos_idx] = -1
            
            return False
        
        # Use sequential variable ordering for simplicity
        if backtrack(0):
            # Count distinct colors used
            distinct = len(set(colors))
            if best_result is None or distinct < best_result[0]:
                best_result = (distinct, colors.copy())
            
            if k == 3:  # Can't do better than 3
                break
    
    # If no solution found with backtracking, fall back to greedy
    if best_result is None:
        colors = []
        for x in range(1, 129):
            found = False
            for color in range(128):
                valid = True
                # Check all 4-term APs ending at position x-1 (0-indexed)
                for d in range(1, (x - 1) // 3 + 1):
                    a = x - 1 - 3 * d
                    if a < 0:
                        break
                    b = a + d
                    c = a + 2 * d
                    prog_colors = {colors[a], colors[b], colors[c], color}
                    if len(prog_colors) < 3:
                        valid = False
                        break
                if valid:
                    colors.append(color)
                    found = True
                    break
            if not found:
                # Shouldn't happen with 128 colors, but fallback
                colors.append(0)
        best_result = (len(set(colors)), colors)
    
    return best_result[1]
# EVOLVE-BLOCK-END
