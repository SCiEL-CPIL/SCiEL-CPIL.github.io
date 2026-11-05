import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
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
    # This serves as a fallback and a baseline.
    greedy_colors = []
    for x in range(1, n + 1):
        idx = x - 1
        best_color = -1
        for color in range(128):
            valid = True
            # Check 4-term APs ending at idx
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
            # Check 3-term APs ending at idx (necessary condition for 4-term APs involving this pos as last element? 
            # Actually, if we assign color to idx, we must ensure that any 3-term AP ending at idx 
            # doesn't force a violation in a 4-term AP? 
            # The seed code checks: all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))
            # This checks if color == c_{idx-d} == c_{idx-2d}. If so, then in AP (idx-3d, idx-2d, idx-d, idx), 
            # if idx-3d is also color, it's 1 color. If not, it's 2 colors. 
            # Wait, the seed logic is a bit heuristic. Let's stick to the full 4-AP check for safety in greedy.
            # But for speed, the seed's 3-term check is a good pruning.
            # If c_{idx-d} == c_{idx-2d} == color, then we have 3 same. 
            # The 4th element is idx-3d. 
            # If c_{idx-3d} is also color, set size 1. Invalid.
            # If c_{idx-3d} is different, set size 2. Invalid.
            # So if c_{idx-d} == c_{idx-2d} == color, it is ALWAYS invalid for the AP ending at idx with step d?
            # Yes, because the AP is (idx-3d, idx-2d, idx-d, idx).
            # Values: c_{idx-3d}, color, color, color.
            # Distinct colors: {c_{idx-3d}, color}. Size is 1 or 2. Always < 3.
            # So this is a valid pruning condition.
            for d in range(1, (idx // 2) + 1):
                if idx - 2 * d >= 0:
                    if greedy_colors[idx - d] == color and greedy_colors[idx - 2 * d] == color:
                        valid = False
                        break
            if valid:
                best_color = color
                break
        if best_color == -1:
            # Should not happen with 128 colors
            best_color = 0
        greedy_colors.append(best_color)
        
    num_greedy_colors = len(set(greedy_colors))
    
    # If greedy uses 3 colors, it's optimal.
    if num_greedy_colors <= 3:
        return greedy_colors
        
    # Try backtracking for K=3, 4, 5
    # We will try to find a solution with minimal K.
    
    def solve_with_k_colors(k):
        if time.time() > deadline:
            return None
        
        colors = [-1] * n
        
        # State for each AP:
        # distinct_colors: set of colors assigned so far in this AP
        # unassigned_count: number of unassigned positions in this AP
        ap_distinct = [set() for _ in range(len(ap_list))]
        ap_unassigned = [4] * len(ap_list)
        
        # Initialize: all unassigned, distinct empty
        # We don't need to initialize explicitly if we compute on the fly or maintain state.
        # Maintaining state is faster.
        
        def is_consistent(pos, color):
            """
            Check if assigning 'color' to 'pos' is consistent with current assignments.
            Returns True if consistent, False otherwise.
            """
            for ap_idx in pos_to_aps[pos]:
                a, b, c, d_pos = ap_list[ap_idx]
                # Get current colors in this AP
                # We need to check if this assignment leads to a violation.
                # A violation occurs if the final set of colors has size < 3.
                # Let S be the set of colors in the AP after assignment.
                # Let U be the set of unassigned positions in the AP after assignment.
                # If |S| >= 3, it's safe (any assignment to U will keep size >= 3? 
                # Wait. If |S|=3, and we add a 4th color, size becomes 3 or 4. Always >= 3. Safe.)
                # If |S| == 2, we need at least one position in U to be assigned a color NOT in S.
                # If |S| == 1, we need at least two positions in U to be assigned colors NOT in S (and distinct from each other? No, just distinct from S and each other? 
                # If S={A}, U has 3 spots. We need final size >= 3. 
                # If we assign B, B, B. Final {A, B}. Size 2. Invalid.
                # If we assign B, C, C. Final {A, B, C}. Size 3. Valid.
                # So we need the set of colors assigned to U to have size >= 3 - |S|? 
                # Actually, if |S|=1, we need the colors in U to contribute at least 2 new distinct colors.
                # If |S|=2, we need the colors in U to contribute at least 1 new distinct color.
                
                # Let's compute S and U for this AP given the trial assignment.
                vals = [colors[a], colors[b], colors[c], colors[d_pos]]
                # pos is one of a,b,c,d_pos
                idx_in_ap = a == pos
                if not idx_in_ap:
                    idx_in_ap = b == pos
                if not idx_in_ap:
                    idx_in_ap = c == pos
                if not idx_in_ap:
                    idx_in_ap = d_pos == pos
                
                # Replace pos's value with color
                old_val = vals[idx_in_ap]
                vals[idx_in_ap] = color
                
                S = set(v for v in vals if v != -1)
                U_count = sum(1 for v in vals if v == -1)
                
                if len(S) >= 3:
                    continue # Safe
                elif len(S) == 2:
                    if U_count == 0:
                        return False # Already violated
                    # If U_count >= 1, is it possible to satisfy?
                    # We need at least one new color from U.
                    # Since we are assigning colors from 0..k-1, and k>=3, there is always a color not in S.
                    # So it is POSSIBLE. We don't prune here, but we might in forward checking.
                    pass
                elif len(S) == 1:
                    if U_count == 0:
                        return False
                    if U_count == 1:
                        # Only 1 spot left. It can provide at most 1 new color.
                        # Final size will be 2. Invalid.
                        return False
                    # If U_count >= 2, possible.
                    pass
                else: # len(S) == 0
                    if U_count == 0:
                        return False
                    # If U_count < 3, can we get 3 distinct colors?
                    # If U_count == 2, max distinct is 2. Invalid.
                    if U_count < 3:
                        return False
                    # If U_count == 3, max distinct is 3. Possible.
                    pass
            return True

        def forward_check(pos, color):
            """
            Check if any unassigned variable has an empty domain.
            """
            # For each unassigned position j, check if there is any color c in 0..k-1
            # such that is_consistent(j, c) is True.
            # This is expensive. 
            # Alternative: just check if the assignment to 'pos' causes any AP to become unsatisfiable 
            # regardless of future assignments.
            # This is what is_consistent does partially.
            # A stronger check: for each AP containing pos, check if the constraint can still be met.
            # We did that in is_consistent.
            # Let's do a light forward check: for each neighbor j of pos (in same AP), 
            # if j is unassigned, check if it has at least one valid color.
            
            for ap_idx in pos_to_aps[pos]:
                a, b, c, d_pos = ap_list[ap_idx]
                neighbors = [a, b, c, d_pos]
                neighbors.remove(pos)
                for j in neighbors:
                    if colors[j] == -1:
                        # Check if j has any valid color
                        has_valid = False
                        for c_j in range(k):
                            if is_consistent(j, c_j):
                                has_valid = True
                                break
                        if not has_valid:
                            return False
            return True

        def backtrack(depth):
            if time.time() > deadline:
                return False
            
            if depth == n:
                return True
            
            # MRV: Find unassigned variable with smallest domain
            best_var = -1
            best_domain_size = k + 1
            for i in range(n):
                if colors[i] == -1:
                    valid_count = 0
                    for c in range(k):
                        if is_consistent(i, c):
                            valid_count += 1
                    if valid_count < best_domain_size:
                        best_domain_size = valid_count
                        best_var = i
                        if best_domain_size == 0:
                            return False # Dead end
            
            if best_var == -1:
                return True # All assigned

            for c in range(k):
                if is_consistent(best_var, c):
                    colors[best_var] = c
                    if forward_check(best_var, c):
                        if backtrack(depth + 1):
                            return True
                    colors[best_var] = -1
            
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

    # Fallback to greedy
    return greedy_colors
