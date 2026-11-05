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
        # count[color]: number of times color is assigned in this AP
        # assigned_count: number of positions assigned in this AP
        # distinct_count: number of distinct colors assigned in this AP
        ap_count = [[0]*k for _ in range(len(ap_list))]
        ap_assigned = [0] * len(ap_list)
        ap_distinct = [0] * len(ap_list)
        
        def is_valid_ap(ap_idx):
            """
            Check if the current state of this AP is not already invalid,
            and if it is still possible to satisfy the constraint.
            Constraint: Final distinct colors >= 3.
            """
            d = ap_distinct[ap_idx]
            a = ap_assigned[ap_idx]
            unassigned = 4 - a
            
            # If already 3 distinct, it's safe (adding more colors keeps it >= 3)
            if d >= 3:
                return True
            
            # If all assigned, check if distinct >= 3
            if unassigned == 0:
                return d >= 3
            
            # Potential distinct colors = current distinct + unassigned (max)
            # But we can only add colors from 0..k-1.
            # If k < 3, impossible. But we call with k>=3.
            
            # If current distinct + unassigned < 3, impossible.
            if d + unassigned < 3:
                return False
            
            # Additional check for K=3:
            # If we have 2 distinct colors, and 1 unassigned, we MUST use the 3rd color.
            # If we have 1 distinct color, and 2 unassigned, we MUST use 2 new distinct colors.
            # This is covered by d + unassigned >= 3.
            
            # However, we can prune earlier if the counts force a violation?
            # No, the "potential" check is sufficient for feasibility.
            # The "violation" check is: if d < 3 and unassigned == 0, invalid.
            # Which is covered above.
            
            return True

        def assign(pos, color):
            for ap_idx in pos_to_aps[pos]:
                ap_count[ap_idx][color] += 1
                if ap_count[ap_idx][color] == 1:
                    ap_distinct[ap_idx] += 1
                ap_assigned[ap_idx] += 1

        def unassign(pos, color):
            for ap_idx in pos_to_aps[pos]:
                ap_count[ap_idx][color] -= 1
                if ap_count[ap_idx][color] == 0:
                    ap_distinct[ap_idx] -= 1
                ap_assigned[ap_idx] -= 1

        def backtrack(depth):
            if time.time() > deadline:
                return False
            
            if depth == n:
                return True
            
            # MRV: Find unassigned variable with smallest domain
            # To speed up, we can just pick the next unassigned position in order,
            # or use a heuristic.
            # For N=128, simple order might be okay if pruning is strong.
            # Let's try simple order first for speed of MRV calculation.
            # Actually, MRV is better. But calculating domain size for all unassigned is O(N * K * AvgAPs).
            # Let's try a hybrid: pick the first unassigned position, but if it fails, backtrack.
            # Or, pick the position with the most constraints?
            
            # Let's stick to simple order 0..n-1 for now, as it's fast to iterate.
            # If it's too slow, we can switch to MRV.
            pos = depth
            if colors[pos] != -1:
                return backtrack(depth + 1)
            
            # Try colors in order
            # Heuristic: Try colors that are less frequent? Or just 0..k-1
            for c in range(k):
                # Check if assigning c to pos is consistent
                valid = True
                # We need to check all APs containing pos
                for ap_idx in pos_to_aps[pos]:
                    # Simulate assignment
                    # We can't just check is_valid_ap because we haven't updated state yet.
                    # We need to check if the state AFTER assignment is valid.
                    
                    # Current state of ap_idx:
                    d = ap_distinct[ap_idx]
                    a = ap_assigned[ap_idx]
                    cnt = ap_count[ap_idx][c]
                    
                    # New state:
                    new_a = a + 1
                    new_d = d
                    if cnt == 0:
                        new_d = d + 1
                    new_unassigned = 4 - new_a
                    
                    # Check if new state is feasible
                    if new_d >= 3:
                        continue # Safe
                    if new_unassigned == 0:
                        if new_d < 3:
                            valid = False
                            break
                        else:
                            continue
                    if new_d + new_unassigned < 3:
                        valid = False
                        break
                    # If k==3, and new_d==2, new_unassigned==1, we are forced to use the 3rd color.
                    # This doesn't invalidate the current step, but restricts future steps.
                    # Our feasibility check handles this.
                
                if valid:
                    colors[pos] = c
                    assign(pos, c)
                    if backtrack(depth + 1):
                        return True
                    unassign(pos, c)
                    colors[pos] = -1
            
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
