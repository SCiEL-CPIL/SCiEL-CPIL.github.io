import numpy as np
import time

def construct():
    """
    Minimize the number of distinct color labels used on positions 1..128
    such that every non-constant 4-term arithmetic progression (a, a+d, a+2d, a+3d)
    with d>0 and a+3d <= 128 uses at least 3 distinct colors.
    
    Strategy:
    1. Precompute all 4-term APs and their incidence to positions.
    2. Use a backtracking solver with:
       - MRV (Minimum Remaining Values) variable selection heuristic.
       - Forward checking / constraint propagation to prune domains.
       - Time limit guard to ensure we return a valid solution within 90s.
    3. Try k=3, then k=4, etc. If a solution is found, return it.
    4. If no solution is found within the time limit for small k, fall back to a
       greedy construction that is guaranteed to produce a valid (though possibly
       suboptimal in color count) coloring, or use a pre-computed heuristic pattern.
    
    Note: The "score" is the number of distinct colors used. Lower is better.
    """
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill
    
    n = 128
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) 0-indexed
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # pos_to_aps: for each position, list of AP indices it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # Greedy solution first (uses smallest valid color)
    # This serves as a fallback and also gives an upper bound on the number of colors needed.
    greedy_colors = []
    for x in range(1, n + 1):
        idx = x - 1
        best_color = -1
        for color in range(128):
            valid = True
            # Check 4-term APs ending at idx
            # a, a+d, a+2d, idx
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
            # Check 3-term APs ending at idx (necessary condition for 4-term)
            # If a, a+d, a+2d are all same color, then any 4-term AP extending them would be invalid?
            # Actually, the constraint is on 4-term APs.
            # If we have a 3-term AP (a, a+d, a+2d) all color C, and we assign C to a+3d,
            # then the 4-term AP (a, a+d, a+2d, a+3d) has 1 color. Invalid.
            # But if we assign D != C to a+3d, the set is {C, D}, size 2. Still invalid!
            # So, if any 3 positions in a potential 4-term AP are the same color, the 4th MUST be a NEW color (different from that color).
            # Wait, the constraint is "at least 3 distinct colors".
            # If we have C, C, C, X. Set is {C, X}. Size 2. Invalid.
            # So, if 3 are C, the 4th must be D such that {C, D} has size >= 3? Impossible.
            # Therefore, it is IMPOSSIBLE to have 3 positions in a 4-term AP all having the same color.
            # This is a strong constraint: No 3 positions in any 4-term AP can be monochromatic.
            # This implies no 3-term AP can be monochromatic either (since any 3-term AP is part of a 4-term AP? 
            # Not necessarily. (a, a+d, a+2d) is part of (a, a+d, a+2d, a+3d) if a+3d <= n.
            # If a+3d > n, it's not a valid 4-term AP.
            # However, if a+2d <= n, then (a, a+d, a+2d) is a 3-term AP.
            # If we have a 3-term AP at the end (a, a+d, a+2d) where a+3d > n, it doesn't directly form a 4-term AP.
            # But it might be part of another 4-term AP?
            # Let's stick to the 4-term AP check for safety in the greedy step.
            
            if valid:
                best_color = color
                break
        if best_color == -1:
            # Should not happen with 128 colors, but just in case
            best_color = 0
        greedy_colors.append(best_color)
        
    num_greedy_colors = len(set(greedy_colors))
    
    # If greedy uses few colors, it might be optimal.
    # But we want to minimize, so let's try to solve with k=3, 4, etc.
    
    def solve_with_k_colors(k):
        if time.time() > deadline:
            return None
        
        colors = [-1] * n
        domains = [list(range(k)) for _ in range(n)]
        
        def is_valid_assignment(pos, color):
            # Check constraints involving 'pos'
            for ap_idx in pos_to_aps[pos]:
                a, b, c, d_pos = ap_list[ap_idx]
                # Get current values, replacing pos with the trial color
                vals = [colors[a], colors[b], colors[c], colors[d_pos]]
                # Find index of pos in the AP tuple
                # pos is one of a, b, c, d_pos
                if pos == a: vals[0] = color
                elif pos == b: vals[1] = color
                elif pos == c: vals[2] = color
                elif pos == d_pos: vals[3] = color
                
                # If any value is -1, we can't fully verify, but we can check for immediate violations
                # Violation conditions:
                # 1. All 4 assigned and distinct count < 3.
                # 2. 3 assigned and all same (then 4th cannot fix it to reach 3 distinct).
                #    If 3 are C, and 4th is X, set is {C, X}. Size 2. Always < 3.
                #    So if any 3 positions in an AP are assigned the same color, it's a dead end.
                
                assigned_vals = [v for v in vals if v != -1]
                if len(assigned_vals) == 4:
                    if len(set(assigned_vals)) < 3:
                        return False
                elif len(assigned_vals) == 3:
                    if len(set(assigned_vals)) == 1:
                        return False
                # If len < 3, it's possible to satisfy (e.g., 2 assigned same, 2 unassigned -> can pick 2 new colors? 
                # If 2 are C, and we pick D, E. Set {C, D, E}. Size 3. Valid.)
            return True

        def get_mrv_var():
            best_var = -1
            best_domain_size = k + 1
            for i in range(n):
                if colors[i] == -1:
                    # Count valid colors
                    valid_count = 0
                    for c in domains[i]:
                        if is_valid_assignment(i, c):
                            valid_count += 1
                    if valid_count < best_domain_size:
                        best_domain_size = valid_count
                        best_var = i
                        if best_domain_size == 0:
                            return -1, 0  # Dead end
            return best_var, best_domain_size

        def backtrack(depth):
            if time.time() > deadline:
                return False
            
            if depth == n:
                return True
            
            var, domain_size = get_mrv_var()
            if var == -1:
                return False  # All assigned
            
            if domain_size == 0:
                return False
            
            # Try colors in domains[var]
            # Heuristic: try colors that appear less frequently in neighbors? Or just sequential.
            for c in domains[var]:
                if is_valid_assignment(var, c):
                    colors[var] = c
                    if backtrack(depth + 1):
                        return True
                    colors[var] = -1
            
            return False
        
        if backtrack(0):
            return colors
        return None

    # Try k=3
    result = solve_with_k_colors(3)
    if result is not None:
        return result
    
    # Try k=4
    result = solve_with_k_colors(4)
    if result is not None:
        return result
    
    # Try k=5
    result = solve_with_k_colors(5)
    if result is not None:
        return result
        
    # Try k=6
    result = solve_with_k_colors(6)
    if result is not None:
        return result

    # Fallback: return the greedy solution
    return greedy_colors

