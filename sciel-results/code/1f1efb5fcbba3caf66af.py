import time
import random
import sys

def construct():
    """
    Returns n=128 integer color labels in 0..127.
    Every nonconstant four-term progression of positions must use AT LEAST three distinct colors.
    Minimize the number of labels actually used.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    N = 128
    
    # Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d)
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each position, list of (ap_index, index_in_ap)
    pos_aps = [[] for _ in range(N)]
    for i, ap in enumerate(aps):
        for j, p in enumerate(ap):
            pos_aps[p].append((i, j))

    def solve_3_colors():
        """
        Backtracking solver for 3 colors (0, 1, 2).
        Constraint: Every 4-AP must contain exactly one of each pattern {2,1,1} of colors.
        """
        colors = [-1] * N
        # Track for each AP:
        # count[0], count[1], count[2]
        # unassigned count
        ap_counts = [[0, 0, 0] for _ in range(len(aps))]
        ap_unassigned = [4] * len(aps)
        
        # Order of variables: We can stick to 0..N-1 or use dynamic ordering.
        # Dynamic ordering is better but complex. Let's try static first with strong propagation.
        
        def is_valid_state():
            # Check if any AP is in an impossible state
            for i in range(len(aps)):
                c0, c1, c2 = ap_counts[i]
                un = ap_unassigned[i]
                
                # If all assigned, check if valid
                if un == 0:
                    # Must be pattern 2,1,1
                    counts = [c0, c1, c2]
                    counts.sort()
                    if counts != [1, 1, 2]:
                        return False
                    continue
                
                # If some unassigned
                # Check if it's possible to reach 2,1,1
                # Max count for any color is 2.
                # If any count > 2, invalid.
                if c0 > 2 or c1 > 2 or c2 > 2:
                    return False
                
                # We need to fill 'un' slots to reach total 4.
                # Final counts must be a permutation of (1,1,2).
                # Let target counts be t0, t1, t2.
                # We need t_i >= current_i and sum(t) = 4.
                # Also t_i <= 2.
                
                # Simple check:
                # If current count is 2, we can add 0 more of that color.
                # If current count is 1, we can add 0 or 1 more.
                # If current count is 0, we can add 0, 1, or 2 more.
                
                # Let's see if we can form a valid target.
                # The only valid targets are permutations of (1,1,2).
                # So we need to check if there exists a permutation (t0,t1,t2) of (1,1,2)
                # such that t0 >= c0, t1 >= c1, t2 >= c2.
                
                targets = [(1,1,2), (1,2,1), (2,1,1)]
                possible = False
                for t in targets:
                    if t[0] >= c0 and t[1] >= c1 and t[2] >= c2:
                        possible = True
                        break
                if not possible:
                    return False
            return True

        def backtrack(pos):
            if time.time() > deadline:
                return False
            
            if pos == N:
                return is_valid_state()
            
            # Try colors 0, 1, 2
            for color in range(3):
                # Assign color to pos
                colors[pos] = color
                
                # Update AP counts
                affected_aps = []
                for ap_idx, j in pos_aps[pos]:
                    ap_counts[ap_idx][color] += 1
                    ap_unassigned[ap_idx] -= 1
                    affected_aps.append(ap_idx)
                
                if is_valid_state():
                    if backtrack(pos + 1):
                        return True
                else:
                    # Revert
                    pass
                
                # Revert
                for ap_idx in affected_aps:
                    ap_counts[ap_idx][color] -= 1
                    ap_unassigned[ap_idx] += 1
                colors[pos] = -1
            
            return False
        
        # Initial check
        if not is_valid_state():
            return None
            
        if backtrack(0):
            return colors
        return None

    # Try 3 colors
    res = solve_3_colors()
    if res is not None:
        # Verify
        valid = True
        for ap in aps:
            cs = {res[p] for p in ap}
            if len(cs) < 3:
                valid = False
                break
        if valid:
            return res

    # Fallback: 4 colors
    # We can use a greedy approach for 4 colors, which is very likely to succeed.
    # Or a simple backtracking.
    # Given time, a greedy with backtracking for 4 colors is safer.
    
    def solve_4_colors():
        colors = [-1] * N
        # Greedy with backtracking
        # To speed up, we can just use a greedy that tries 0,1,2,3 and checks validity.
        # If it fails, we might need to backtrack.
        # But 4 colors is very loose.
        
        def check_valid_partial(pos, color):
            # Check all APs containing pos
            for ap_idx, j in pos_aps[pos]:
                ap = aps[ap_idx]
                # Temporarily assign
                old = colors[pos]
                colors[pos] = color
                
                # Check if this AP is now invalid or impossible
                # If all assigned, check distinct >= 3
                # If some unassigned, check if possible to reach >= 3
                distinct = set()
                uncolored = 0
                for p in ap:
                    if colors[p] == -1:
                        uncolored += 1
                    else:
                        distinct.add(colors[p])
                
                colors[pos] = old
                
                if uncolored == 0:
                    if len(distinct) < 3:
                        return False
                else:
                    # Max distinct possible = len(distinct) + uncolored (if we use new colors)
                    # But we only have 4 colors.
                    # So max distinct = min(4, len(distinct) + uncolored)
                    # We need >= 3.
                    if min(4, len(distinct) + uncolored) < 3:
                        return False
            return True

        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == N:
                return True
            
            for color in range(4):
                if check_valid_partial(pos, color):
                    colors[pos] = color
                    if backtrack(pos + 1):
                        return True
                    colors[pos] = -1
            return False
        
        if backtrack(0):
            return colors
        return None

    res4 = solve_4_colors()
    if res4 is not None:
        valid = True
        for ap in aps:
            cs = {res4[p] for p in ap}
            if len(cs) < 3:
                valid = False
                break
        if valid:
            return res4

    # Final Fallback: Identity (128 colors)
    return list(range(128))
