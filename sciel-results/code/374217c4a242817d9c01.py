import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
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
            
    # Greedy solution first
    greedy_colors = []
    for x in range(1, n + 1):
        # x is 1-indexed in loop, but we use 0-indexed for array
        # Wait, seed uses x in 1..128 and colors[x-1]
        # Let's stick to 0-indexed for internal logic
        idx = x - 1
        best_color = -1
        for color in range(128):
            valid = True
            # Check 4-term APs ending at idx
            # a, a+d, a+2d, a+3d = idx
            # a+3d = idx => a = idx - 3d
            # d can be 1..floor(idx/3)
            for d in range(1, (idx // 3) + 1):
                a = idx - 3 * d
                # Positions: a, a+d, a+2d, idx
                c1 = greedy_colors[a]
                c2 = greedy_colors[a + d]
                c3 = greedy_colors[a + 2 * d]
                # Check if {color, c1, c2, c3} has size >= 3
                if len({color, c1, c2, c3}) < 3:
                    valid = False
                    break
            if not valid:
                continue
            # Check 3-term APs ending at idx (necessary condition)
            # a, a+d, a+2d = idx
            # a+2d = idx => a = idx - 2d
            # d can be 1..floor(idx/2)
            for d in range(1, (idx // 2) + 1):
                a = idx - 2 * d
                c1 = greedy_colors[a]
                c2 = greedy_colors[a + d]
                # If color == c1 == c2, invalid
                if color == c1 and c1 == c2:
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
    
    # If greedy uses 3 colors, it's likely optimal (min is 3).
    # If greedy uses > 3, try to find a 3-coloring.
    if num_greedy_colors <= 3:
        return greedy_colors
        
    # Try backtracking for K=3
    # If 3 fails, try K=4? 
    # Given time constraints, we try K=3 first.
    
    colors = [-1] * n
    domains = [list(range(3)) for _ in range(n)] # 0, 1, 2
    
    def is_valid_assignment(pos, color):
        # Check constraints involving 'pos'
        for ap_idx in pos_to_aps[pos]:
            a, b, c, d_pos = ap_list[ap_idx]
            vals = [colors[a], colors[b], colors[c], colors[d_pos]]
            # Replace pos with color
            # Actually, we need to check if the constraint is satisfied or violated
            # If any is -1, we can't fully verify, but we can check if it's ALREADY violated
            # Or if it becomes impossible.
            # Simple check: if all 4 are assigned, verify.
            # If not all assigned, we can check if the assigned ones force a violation.
            # For 4-AP >= 3 colors:
            # Violation if:
            # 1. All assigned and size < 3.
            # 2. If 3 are assigned and they are all same (size 1), then the 4th MUST be different, but even then size is 2. 
            #    Wait, if 3 are 0,0,0, and 4th is x. Set is {0, x}. Size 2. Always invalid.
            #    So if any 3 positions in an AP are same color, the AP is invalid regardless of the 4th.
            # 3. If 2 are assigned and same (e.g., 0,0), and the other 2 are unassigned.
            #    We need the final set to have size >= 3.
            #    This requires at least one of the unassigned to be a new color, AND the other to be a new color or different?
            #    If we have 0,0, x, y. Set {0, x, y}. Size 3 if x!=y and x!=0 and y!=0.
            #    So it's possible.
            
            # Let's just do the full check if all 4 are assigned.
            # And the "3 same" check.
            assigned_vals = [v for v in vals if v != -1]
            if len(assigned_vals) == 4:
                if len(set(assigned_vals)) < 3:
                    return False
            elif len(assigned_vals) == 3:
                if len(set(assigned_vals)) == 1:
                    return False
            # If len < 3, it's possible to satisfy.
        return True

    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
            
        # MRV: find unassigned variable with smallest domain?
        # For simplicity, use sequential with forward checking.
        # Since we are trying K=3, domains are small.
        
        # Try colors in domains[idx]
        for color in domains[idx]:
            colors[idx] = color
            if is_valid_assignment(idx, color):
                # Forward check
                feasible = True
                for j in range(idx + 1, n):
                    if colors[j] == -1:
                        # Check if domain is empty?
                        # We don't prune domains explicitly here, just check validity of current assignment.
                        # This is weaker than full propagation.
                        pass
                if feasible:
                    if backtrack(idx + 1):
                        return True
            colors[idx] = -1
            
        return False

    # Try to solve with 3 colors
    # Use a slightly smarter ordering: positions with more constraints first?
    # For now, sequential.
    if backtrack(0):
        return colors
        
    # If 3 colors failed, return greedy
    return greedy_colors
