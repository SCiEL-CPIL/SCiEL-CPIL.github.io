import time

def construct():
    deadline = time.time() + 115.0
    
    target_diffs = set(range(1, 31))
    full_mask = (1 << 30) - 1

    def is_valid(S):
        diffs = set()
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return target_diffs.issubset(diffs)

    def search(D):
        S = [0]
        used_diffs = set()
        covered_mask = 0
        
        def dfs(start_idx, S, used_diffs, covered_mask):
            if time.time() > deadline:
                return None
            
            if (covered_mask & full_mask) == full_mask:
                return list(S)
                
            missing = full_mask & ~covered_mask
            if missing == 0:
                return list(S)
                
            max_add = 32 - len(S)
            if max_add == 0:
                return None
                
            candidates = []
            # Optimization: Iterate x and check validity
            # To speed up, we can pre-filter x based on simple constraints?
            # For now, direct check.
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
                coverage_count = bin(new_missing_covered).count('1')
                
                # Heuristic: Prefer smaller x if coverage is same, 
                # but prefer higher coverage if x is not too large.
                # Since we want to minimize D, smaller x is generally better for future steps.
                candidates.append((coverage_count, x))
                
            # Sort by coverage (desc), then x (asc)
            candidates.sort(key=lambda t: (-t[0], t[1]))
            
            # Increase candidate limit for better search
            # If D is small, branching is lower.
            limit = 500 if D < 100 else 200
            if len(candidates) > limit:
                candidates = candidates[:limit]
                
            for _, x in candidates:
                new_used = set(used_diffs)
                new_mask = covered_mask
                for s in S:
                    d = x - s
                    new_used.add(d)
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                        
                S.append(x)
                res = dfs(start_idx + 1, S, new_used, new_mask)
                if res is not None:
                    return res
                S.pop()
                
            return None
        
        return dfs(0, S, used_diffs, 0)

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
