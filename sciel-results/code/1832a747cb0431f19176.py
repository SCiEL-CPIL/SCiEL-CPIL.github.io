import time

def construct():
    deadline = time.time() + 115.0
    
    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1
    popcount = [0] * (1 << 30)
    for i in range(1, 1 << 30):
        popcount[i] = popcount[i >> 1] + (i & 1)

    def search(D):
        S = [0]
        used_diffs = set()
        covered_mask = 0
        # Precompute: for each x, which existing S elements make x-s in used_diffs?
        # We'll do this dynamically.
        
        def dfs(S, used_diffs, covered_mask):
            if time.time() > deadline:
                return None
            
            if (covered_mask & full_mask) == full_mask:
                return list(S)
                
            missing = full_mask & ~covered_mask
            missing_count = popcount[missing]
                
            max_add = 32 - len(S)
            if max_add == 0:
                return None
            
            # Pruning: each new element can cover at most len(S) new differences (differences to existing elements)
            # But we need to cover missing_count differences.
            # A new element x can cover at most min(len(S), missing_count) new target differences.
            if max_add * len(S) < missing_count:
                return None
            
            candidates = []
            # Optimization: Iterate x and check validity
            for x in range(S[-1] + 1, D + 1):
                ok = True
                new_mask = 0
                for s in S:
                    d = x - s
                    if d in used_diffs:
                        ok = False
                        break
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                
                if not ok:
                    continue
                
                new_missing_covered = new_mask & missing
                coverage_count = popcount[new_missing_covered]
                
                # If this doesn't cover any new target diff and we have few elements left, skip
                # Actually, we might still need it for future coverage, so be careful
                # But if coverage_count == 0 and we have limited budget, it's risky
                candidates.append((coverage_count, x))
                
            if not candidates:
                return None
            
            # Sort by coverage (desc), then x (asc)
            candidates.sort(key=lambda t: (-t[0], t[1]))
            
            # Limit candidates to control branching
            limit = 300 if D < 100 else 150
            if len(candidates) > limit:
                candidates = candidates[:limit]
                
            for _, x in candidates:
                new_used = used_diffs | set()  # shallow copy
                new_mask = covered_mask
                for s in S:
                    d = x - s
                    new_used.add(d)
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                        
                S.append(x)
                res = dfs(S, new_used, new_mask)
                if res is not None:
                    return res
                S.pop()
                
            return None
        
        return dfs(S, used_diffs, 0)

    # Search for minimum D
    best_res = None
    for D in range(30, 500):
        if time.time() > deadline - 5:
            break
        res = search(D)
        if res is not None:
            return res
            
    # Fallback greedy
    S = [0]
    used = set()
    target = set(range(1, 31))
    
    while target:
        best_x = None
        best_score = -1
        # Search x in a reasonable range
        # Use a wider range for greedy to ensure validity
        search_range = max(S) + 5000
        for x in range(max(S) + 1, search_range):
            ok = True
            cover = 0
            for s in S:
                d = x - s
                if d in used:
                    ok = False
                    break
                if d in target:
                    cover += 1
            if ok:
                score = cover * 1000 - x
                if score > best_score:
                    best_score = score
                    best_x = x
            if time.time() > deadline - 2:
                break
        if best_x is None:
            # If no good x found in heuristic range, try to find ANY valid x
            for x in range(max(S) + 1, search_range + 5000):
                ok = True
                for s in S:
                    if x - s in used:
                        ok = False
                        break
                if ok:
                    best_x = x
                    break
            if time.time() > deadline - 2:
                break
        if best_x is None:
            break
        S.append(best_x)
        for s in S[:-1]:
            used.add(best_x - s)
        target -= used
        
        if time.time() > deadline - 2:
            break
            
    return S
