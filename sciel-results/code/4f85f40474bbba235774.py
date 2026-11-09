import numpy as np

def construct():
    N = 256
    S = set()
    candidates = set(range(1, N + 1))
    
    # Helper to check if adding x to S creates a 4-AP
    def creates_4ap(x, S):
        # x is 1st: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
                return True
        # x is 2nd: x-d, x, x+d, x+2d
        for d in range(1, min(x, (N - x) // 2) + 1):
            if (x - d) in S and (x + d) in S and (x + 2 * d) in S:
                return True
        # x is 3rd: x-2d, x-d, x, x+d
        for d in range(1, min(x // 2, N - x) + 1):
            if (x - 2 * d) in S and (x - d) in S and (x + d) in S:
                return True
        # x is 4th: x-3d, x-2d, x-d, x
        for d in range(1, x // 3 + 1):
            if (x - 3 * d) in S and (x - 2 * d) in S and (x - d) in S:
                return True
        return False

    # Helper to calculate gain in 3-APs if x is added to S
    def calc_gain(x, S):
        gain = 0
        # x is 1st: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            if (x + d) in S and (x + 2 * d) in S:
                gain += 1
        # x is 2nd: x-d, x, x+d
        for d in range(1, min(x, N - x) + 1):
            if (x - d) in S and (x + d) in S:
                gain += 1
        # x is 3rd: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            if (x - 2 * d) in S and (x - d) in S:
                gain += 1
        return gain

    # Helper to count how many other candidates form a 4-AP with x
    def count_conflicts_with_candidates(x, candidates):
        count = 0
        # Check 4-APs involving x and another candidate y (where y not in S)
        # This is slightly expensive O(|C| * N), but |C| <= 256, N=256. 256*256*4 ~ 260k ops per check.
        # We do this for each candidate in the loop. Total ~ 256 * 260k = 66M ops.
        # Might be too slow in Python if done naively inside the main loop.
        # Let's optimize: Only do this for tie-breakers?
        # Or just use a simpler heuristic: prefer smaller x?
        # Let's try to optimize the main loop first.
        
        # Actually, we can precompute 4-AP conflicts between all pairs?
        # There are 256*256 pairs. For each pair, check if they are in a 4-AP?
        # A 4-AP is defined by 4 points.
        # Let's stick to the simple tie-breaker first to ensure speed, then try to improve.
        # The current score is 1.18. 
        # Let's try a different tie breaker: prefer x that has higher "potential" 
        # i.e. more 3-APs with OTHER candidates? No, that's what gain is for S.
        
        # Let's just implement the conflict count but efficiently.
        # We only need to know if x and y form a 4-AP.
        # A pair (x, y) forms a 4-AP if there exist a, b such that {a, x, y, b} or {x, a, y, b} etc is a 4-AP.
        # Actually, if x and y are in S, they don't form a 4-AP.
        # We want to know if adding x prevents y from being added later.
        # This happens if x and y are part of a 4-AP with two other elements (which could be in S or not? No, if the other 2 are in S, x is already invalid. If the other 2 are not in S, x and y conflict).
        # So we count pairs (x, y) in candidates such that x, y are in a 4-AP.
        
        # Efficient check for 4-AP between x and y:
        # There are limited ways two points can be in a 4-AP.
        # Positions: (1,2), (1,3), (1,4), (2,3), (2,4), (3,4).
        # If x < y:
        # 1. x, y are 1,2: x, y, y+(y-x), y+2(y-x). Need y+(y-x) and y+2(y-x) to be in candidates or S? 
        #    For conflict, we just need the existence of a 4-AP containing x and y.
        #    The other two elements can be anything in [1, N].
        #    So x and y conflict if there exist u, v in [1, N] such that {x, y, u, v} is a 4-AP.
        #    This is true if the distance d = y-x allows a 4-AP.
        #    Cases:
        #    - x, y are adjacent in AP: d. AP: x, y, x+2d, x+3d. Valid if x+3d <= N.
        #    - x, y are 1,3: 2d. AP: x-2d, x, x+2d, x+4d? No. x, x+d, y, y+d. Valid if x-d>=1 and y+d<=N.
        #    - x, y are 1,4: 3d. AP: x, x+d, x+2d, y. Valid if x+2d < y. (Always true for d=y/3).
        #    - x, y are 2,3: d. AP: x-d, x, y, y+d. Valid if x-d>=1 and y+d<=N.
        #    - x, y are 2,4: 2d. AP: x-2d, x, x+2d, y. Valid if x-2d>=1.
        #    - x, y are 3,4: d. AP: x-2d, x-d, x, y. Valid if x-2d>=1.
        
        # So x and y conflict if ANY of these conditions hold.
        
        return count

    while candidates:
        valid_candidates = []
        for x in candidates:
            if not creates_4ap(x, S):
                valid_candidates.append(x)
                
        if not valid_candidates:
            break
            
        # Calculate scores
        scores = {}
        for x in valid_candidates:
            g = calc_gain(x, S)
            scores[x] = g
            
        # Find max gain
        max_gain = max(scores.values())
        best_candidates = [x for x, g in scores.items() if g == max_gain]
        
        if len(best_candidates) == 1:
            best_x = best_candidates[0]
        else:
            # Tie-breaker: choose x that minimizes conflicts with other valid candidates
            # To save time, just pick the one with smallest index? 
            # Or compute conflicts only for the tie candidates.
            def conflict_count(x):
                cnt = 0
                for y in best_candidates:
                    if x == y: continue
                    # Check if x, y form a 4-AP
                    d = abs(x - y)
                    if d == 0: continue
                    # Check the 6 cases
                    # 1. x, y are 1,2 (x<y): x, y, x+2d, x+3d. Need x+3d <= N
                    if x < y:
                        if x + 3*d <= N: return 1 # Conflict
                        # 2. x, y are 1,3: x, x+d, y, y+d. Need x-d >= 1? No, x is 1st. x, x+d, x+2d, x+3d. 
                        #    If x, y are 1,3, then y = x+2d. AP: x, x+d, y, y+d. 
                        #    Valid if x+d >= 1 (true) and y+d <= N.
                        if y + d <= N: return 1
                        # 3. x, y are 1,4: x, x+d, x+2d, y. y=x+3d. Valid always if d integer.
                        if y == x + 3*d: return 1
                        # 4. x, y are 2,3: x-d, x, y, y+d. y=x+d. Valid if x-d>=1 and y+d<=N.
                        if y == x + d:
                             if x - d >= 1 and y + d <= N: return 1
                        # 5. x, y are 2,4: x-2d, x, x+2d, y. y=x+2d. Valid if x-2d>=1.
                        if y == x + 2*d:
                             if x - 2*d >= 1: return 1
                        # 6. x, y are 3,4: x-2d, x-d, x, y. y=x+d. Valid if x-2d>=1.
                        #    Covered by 4? No, 4 is 2,3. 6 is 3,4.
                        #    If y=x+d, it's case 4 or 6.
                        #    Case 4: x is 2nd. x-d, x, x+d, x+2d.
                        #    Case 6: x is 3rd. x-2d, x-d, x, x+d.
                        #    Both require x-2d >= 1 for case 6. Case 4 requires x-d>=1.
                        #    If x-d>=1, case 4 is possible? We need x+2d<=N.
                        #    So if y=x+d, check if (x-d>=1 and x+2d<=N) OR (x-2d>=1).
                        #    My previous check for case 4 was `x-d>=1 and y+d<=N` which is `x+2d<=N`.
                        #    So I need to add case 6 check: `x-2d>=1`.
                        if y == x + d:
                             if x - 2*d >= 1: return 1
                    else: # y < x
                        # Symmetric
                        # 1. y, x are 1,2: y, x, y+2d, y+3d. Need y+3d <= N.
                        if y + 3*d <= N: return 1
                        # 2. y, x are 1,3: y, y+d, x, x+d. Need x+d <= N.
                        if x + d <= N: return 1
                        # 3. y, x are 1,4: y, y+d, y+2d, x.
                        if x == y + 3*d: return 1
                        # 4. y, x are 2,3: y-d, y, x, x+d. Need y-d>=1 and x+d<=N.
                        if x == y + d:
                             if y - d >= 1 and x + d <= N: return 1
                        # 5. y, x are 2,4: y-2d, y, y+2d, x. Need y-2d>=1.
                        if x == y + 2*d:
                             if y - 2*d >= 1: return 1
                        # 6. y, x are 3,4: y-2d, y-d, y, x. Need y-2d>=1.
                        if x == y + d:
                             if y - 2*d >= 1: return 1
                return cnt
            
            # This is getting complex and might be slow.
            # Let's just pick the smallest x in tie-breaker for now, as it's safe and fast.
            # Or maybe largest x?
            # Let's try smallest x.
            best_x = min(best_candidates)
            
        S.add(best_x)
        candidates.discard(best_x)
        
    return sorted(S)
