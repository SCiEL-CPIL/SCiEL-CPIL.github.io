import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 128
    
    # Precompute APs and index by position
    aps = []
    ap_ids_by_pos = [[] for _ in range(N)]
    
    ap_id = 0
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            p1 = a - 1
            p2 = a + d - 1
            p3 = a + 2 * d - 1
            p4 = a + 3 * d - 1
            ap = (p1, p2, p3, p4)
            aps.append(ap)
            ap_ids_by_pos[p1].append(ap_id)
            ap_ids_by_pos[p2].append(ap_id)
            ap_ids_by_pos[p3].append(ap_id)
            ap_ids_by_pos[p4].append(ap_id)
            ap_id += 1
            
    def solve(k):
        colors = [-1] * N
        timed_out = False
        
        def is_valid_assignment(pos, color):
            # Check only APs containing 'pos'
            for ap_id in ap_ids_by_pos[pos]:
                p1, p2, p3, p4 = aps[ap_id]
                c1 = colors[p1]
                c2 = colors[p2]
                c3 = colors[p3]
                c4 = colors[p4]
                
                if c1 == -1 or c2 == -1 or c3 == -1 or c4 == -1:
                    continue
                
                distinct = len({c1, c2, c3, c4})
                if distinct < 3:
                    return False
            return True

        def backtrack(idx):
            nonlocal timed_out
            if time.time() > deadline:
                timed_out = True
                return False
            
            if idx == N:
                return True
            
            # Heuristic: Try colors in order 0..k-1
            for c in range(k):
                colors[idx] = c
                if is_valid_assignment(idx, c):
                    if backtrack(idx + 1):
                        return True
                colors[idx] = -1
                if timed_out:
                    return False
            
            return False

        if backtrack(0):
            return colors
        return None

    # Try k=3
    res = solve(3)
    if res is not None:
        return res
        
    # Try k=4
    res = solve(4)
    if res is not None:
        return res

    # Fallback greedy
    colors = []
    for x in range(1, N + 1):
        pos = x - 1
        found = False
        for c in range(128):
            valid = True
            for ap_id in ap_ids_by_pos[pos]:
                p1, p2, p3, p4 = aps[ap_id]
                # In greedy, we assign sequentially, so if p > pos, it's unassigned (-1)
                # But we are building 'colors' list, so we can check
                c1 = colors[p1] if p1 < pos else -1
                c2 = colors[p2] if p2 < pos else -1
                c3 = colors[p3] if p3 < pos else -1
                c4 = c
                
                # If any is -1, we can't fully validate yet, but in greedy 
                # we only assign if it doesn't violate completed APs?
                # Actually, greedy usually checks if the assignment is consistent 
                # with previous assignments.
                # If an AP has unassigned parts, we assume they can be fixed later.
                # So we only check if the assigned parts already force < 3 distinct colors
                # AND there are no unassigned parts to fix it?
                # No, if assigned parts use 1 color, and 1 is unassigned, we can fix it.
                # If assigned parts use 1 color, and 0 are unassigned, it's a violation.
                # If assigned parts use 2 colors, and 0 are unassigned, it's a violation.
                
                # Simple check: if all 4 are assigned (c1,c2,c3 are in colors, c4 is c),
                # check distinct.
                # Since we go left to right, p1,p2,p3 < pos implies they are assigned.
                # p4 > pos implies unassigned.
                
                if c1 != -1 and c2 != -1 and c3 != -1 and c4 != -1:
                    if len({c1, c2, c3, c4}) < 3:
                        valid = False
                        break
            if valid:
                colors.append(c)
                found = True
                break
        if not found:
            colors.append(0)
            
    return colors
