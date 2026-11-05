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
            # Check 3-term APs ending at idx
            for d in range(1, (idx // 2) + 1):
                if idx - 2 * d >= 0:
                    if greedy_colors[idx - d] == color and greedy_colors[idx - 2 * d] == color:
                        valid = False
                        break
            if valid:
                best_color = color
                break
        if best_color == -1:
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
        
        # Precompute bitmasks for each AP: which positions are in it
        # For each AP, store a list of 4 position indices
        ap_positions = ap_list  # already stored as tuples
        
        def is_consistent(pos, color):
            """
            Check if assigning 'color' to 'pos' is consistent with current assignments.
            Returns True if consistent, False otherwise.
            """
            color_bit = 1 << color
            for ap_idx in pos_to_aps[pos]:
                a, b, c, d_pos = ap_positions[ap_idx]
                
                # Build the set of colors in this AP after trial assignment
                # Use bitwise OR to accumulate color bits
                mask = 0
                unassigned_count = 0
                
                for p in (a, b, c, d_pos):
                    if p == pos:
                        mask |= color_bit
                    else:
                        col = colors[p]
                        if col == -1:
                            unassigned_count += 1
                        else:
                            mask |= (1 << col)
                
                # Count distinct colors in mask
                distinct_count = bin(mask).count('1')
                
                if distinct_count >= 3:
                    continue # Safe
                elif distinct_count == 2:
                    if unassigned_count == 0:
                        return False # Already violated
                    pass
                elif distinct_count == 1:
                    if unassigned_count == 0:
                        return False
                    if unassigned_count == 1:
                        return False
                    pass
                else: # distinct_count == 0
                    if unassigned_count < 3:
                        return False
                    pass
            return True

        def forward_check(pos, color):
            """
            Check if any unassigned variable has an empty domain.
            """
            for ap_idx in pos_to_aps[pos]:
                a, b, c, d_pos = ap_positions[ap_idx]
                neighbors = [a, b, c, d_pos]
                neighbors.remove(pos)
                for j in neighbors:
                    if colors[j] == -1:
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
                            return False
            
            if best_var == -1:
                return True

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
