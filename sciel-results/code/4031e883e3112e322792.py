import time

def construct():
    # Deadline: 150s wall clock limit. Leave 5s margin.
    deadline = time.time() + 145.0
    
    target_mask = (1 << 30) - 1  # Bits 0..29 correspond to diffs 1..30
    
    # Precompute masks for differences 1..30
    diff_masks = [0] * 31
    for d in range(1, 31):
        diff_masks[d] = 1 << (d - 1)

    def search(D):
        """
        Find a Sidon set S in [0, D] containing 0 such that 
        {s_j - s_i} covers 1..30.
        Returns the set S or None.
        """
        start_time = time.time()
        
        # State:
        # S: list of selected elements, sorted
        # used_diffs: bitmask of all differences found so far (up to D)
        # covered_mask: bitmask of which target diffs (1..30) are covered
        
        def dfs(S, used_diffs, covered_mask):
            if time.time() > deadline - 1.0:
                return None
            
            # Check success
            if (covered_mask & target_mask) == target_mask:
                return list(S)
            
            # Pruning:
            # 1. Not enough elements left?
            # Max elements we can add is D - S[-1]
            # But we are limited by 32 elements total.
            max_add = min(D - S[-1], 32 - len(S))
            if max_add <= 0:
                return None
            
            missing = target_mask & ~covered_mask
            num_missing = bin(missing).count('1')
            
            # 2. Can we cover the missing diffs?
            # Each new element x adds differences x - s for all s in S.
            # Plus differences between new elements.
            # Upper bound on new distinct diffs:
            # If we add k elements, max new diffs is k * len(S) + k*(k-1)/2.
            # We can add at most max_add elements.
            potential = max_add * len(S) + max_add * (max_add - 1) // 2
            if potential < num_missing:
                return None
            
            # Generate candidates
            cands = []
            last = S[-1]
            for x in range(last + 1, D + 1):
                # Check Sidon property: x - s must not be in used_diffs
                ok = True
                new_mask = 0
                for s in S:
                    d = x - s
                    # Check if d is already used
                    # used_diffs is a bitmask. Bit (d-1) corresponds to diff d.
                    # If d > 0, check bit d-1.
                    if (used_diffs >> (d - 1)) & 1:
                        ok = False
                        break
                    if 1 <= d <= 30:
                        new_mask |= diff_masks[d]
                
                if not ok:
                    continue
                
                # Score: number of NEW missing diffs covered
                new_cover = new_mask & missing
                score = bin(new_cover).count('1')
                
                # Heuristic: Prefer candidates that cover more missing diffs.
                # Tie-break: smaller x (to keep diameter small? No, D is fixed. 
                # Smaller x leaves more room for larger x later? 
                # Actually, we are minimizing D globally. Inside search(D), 
                # any valid set is equally good. But smaller x might allow 
                # more flexibility? Or larger x? 
                # Usually, spreading elements out helps Sidon. 
                # Let's try sorting by score desc, then x asc.)
                cands.append((score, x))
            
            # Sort candidates
            cands.sort(key=lambda t: (-t[0], t[1]))
            
            # Limit candidates to avoid explosion in deep nodes
            # If there are too many, take the top 50
            if len(cands) > 50:
                cands = cands[:50]
            
            for _, x in cands:
                # Update state
                new_used = used_diffs
                new_mask = covered_mask
                for s in S:
                    d = x - s
                    new_used |= (1 << (d - 1))
                    if 1 <= d <= 30:
                        new_mask |= diff_masks[d]
                
                S.append(x)
                res = dfs(S, new_used, new_mask)
                S.pop()
                
                if res is not None:
                    return res
            
            return None

        # Initial state
        S = [0]
        used_diffs = 0
        covered_mask = 0
        
        return dfs(S, used_diffs, covered_mask)

    # Search for minimal D
    # Lower bound:
    # We need 30 diffs.
    # Sidon set of size k has ~k^2/2 diffs.
    # To cover 1..30, we need at least enough diffs.
    # Also, the largest diff is D.
    # Empirically, D starts around 30-40.
    
    best_solution = None
    
    # Try D from 30 upwards
    # If we find a solution for D, we return it immediately 
    # because we iterate D in increasing order.
    for D in range(30, 100):
        if time.time() > deadline - 5.0:
            break
            
        # Allocate a small time slice for each D? 
        # Or just let it run until deadline?
        # If search(D) is slow, it might eat the whole time.
        # We should probably cap the time per D.
        # But DFS is all-or-nothing mostly.
        # Let's just run it. If it times out, it returns None.
        
        res = search(D)
        if res is not None:
            return res
            
    # Fallback if no solution found in [30, 100) within time
    # This should not happen if the code is correct and time is sufficient.
    # But just in case, return a valid set from a greedy construction 
    # or the last best attempt? 
    # The task requires a valid set. 
    # If we didn't find one, we must construct one.
    
    # Greedy fallback (slow but robust)
    S = [0]
    used = set()
    target = set(range(1, 31))
    while target and time.time() < deadline - 1.0:
        best_x = None
        best_cover = -1
        # Search for x in a reasonable range
        max_x = max(S) + 100
        for x in range(max(S) + 1, max_x + 1):
            ok = True
            cover = 0
            for s in S:
                d = x - s
                if d in used:
                    ok = False
                    break
                if d in target:
                    cover += 1
            if ok and cover > best_cover:
                best_cover = cover
                best_x = x
        if best_x is None:
            # If no good x found, just find any valid x
            for x in range(max(S) + 1, max(S) + 500):
                ok = True
                for s in S:
                    if x - s in used:
                        ok = False
                        break
                if ok:
                    best_x = x
                    break
        if best_x is None:
            break
        S.append(best_x)
        for s in S[:-1]:
            used.add(best_x - s)
        target -= used
        
    return S
