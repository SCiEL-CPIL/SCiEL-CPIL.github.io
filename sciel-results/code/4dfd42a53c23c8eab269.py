import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 128
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) 0-indexed
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each position, precompute which APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # Greedy solution first - uses smallest valid color
    greedy_colors = [-1] * n
    for idx in range(n):
        best_color = -1
        for color in range(128):
            valid = True
            # Check 4-term APs where this position is the last element
            for d in range(1, (idx // 3) + 1):
                a = idx - 3 * d
                c1 = greedy_colors[a]
                c2 = greedy_colors[a + d]
                c3 = greedy_colors[a + 2 * d]
                if len({color, c1, c2, c3}) < 3:
                    valid = False
                    break
            if not valid:
                continue
            greedy_colors[idx] = color
            break
        if greedy_colors[idx] == -1:
            greedy_colors[idx] = 0  # Should not happen
    
    num_greedy_colors = len(set(greedy_colors))
    if num_greedy_colors <= 3:
        return greedy_colors
    
    # Try backtracking for K=3 with optimized constraint checking
    def solve_with_k_colors(k, time_limit):
        if time.time() > time_limit:
            return None
        
        colors = [-1] * n
        
        # For each AP, track how many positions are assigned and their values
        # Constraint: at least 3 distinct colors in each AP
        
        # Precompute for each position which APs it's in, and the other 3 positions
        pos_ap_info = [[] for _ in range(n)]
        for ap_idx, (a, b, c, d_pos) in enumerate(ap_list):
            for pos in [a, b, c, d_pos]:
                others = [a, b, c, d_pos]
                others.remove(pos)
                pos_ap_info[pos].append((ap_idx, others))
        
        def is_valid_assignment(pos, color):
            """Check if assigning color to pos violates any constraint."""
            for ap_idx, others in pos_ap_info[pos]:
                vals = [color] + [colors[o] for o in others]
                assigned_vals = [v for v in vals if v != -1]
                if len(assigned_vals) == 4:
                    if len(set(assigned_vals)) < 3:
                        return False
                elif len(assigned_vals) == 3:
                    # If 3 are assigned and all same, impossible to get 3 distinct
                    if len(set(assigned_vals)) == 1:
                        return False
            return True
        
        def backtrack(depth):
            if time.time() > time_limit:
                return False
            
            if depth == n:
                return True
            
            # Find unassigned variable with MRV (minimum remaining values)
            best_var = -1
            best_count = k + 1
            for i in range(n):
                if colors[i] == -1:
                    valid_count = 0
                    for c in range(k):
                        if is_valid_assignment(i, c):
                            valid_count += 1
                    if valid_count < best_count:
                        best_count = valid_count
                        best_var = i
                        if valid_count == 0:
                            return False  # Dead end
            
            if best_var == -1:
                return True  # All assigned
            
            for c in range(k):
                if is_valid_assignment(best_var, c):
                    colors[best_var] = c
                    if backtrack(depth + 1):
                        return True
                    colors[best_var] = -1
            
            return False
        
        if backtrack(0):
            return colors
        return None
    
    # Try K=3
    result = solve_with_k_colors(3, deadline)
    if result is not None:
        return result
    
    # Try K=4
    result = solve_with_k_colors(4, deadline)
    if result is not None:
        return result
    
    # Try K=5
    result = solve_with_k_colors(5, deadline)
    if result is not None:
        return result
    
    # Fallback to greedy
    return greedy_colors
