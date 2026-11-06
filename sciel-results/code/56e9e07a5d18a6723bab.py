import time

def construct():
    deadline = time.time() + 110.0
    
    # Target differences
    target_diffs = set(range(1, 31))
    target_mask = (1 << 30) - 1  # 1..30 set to 1
    
    def mask_for_diff(d):
        if 1 <= d <= 30:
            return 1 << (d - 1)
        return 0

    def search(D):
        # We need to find a set S of size k such that:
        # 1. 0 in S
        # 2. max(S) <= D
        # 3. Sidon property
        # 4. Covers 1..30
        
        # Lower bound on size k:
        # Number of differences is k*(k-1)/2.
        # We need at least 30 differences.
        # k=9 -> 36 diffs.
        # k=8 -> 28 diffs (impossible).
        # So min k is 9.
        
        # We will try to find a solution with k elements.
        # Start with small k and increase if no solution found for D.
        # However, usually smaller k allows larger D? No, smaller k means fewer diffs, so harder to cover 1..30.
        # Actually, for a fixed D, a larger k gives more flexibility to cover differences, 
        # but also more constraints.
        # Let's just search for ANY valid set of size k in [0, D].
        # We can fix the size k? 
        # The problem asks for "at most 32 distinct integers".
        # Minimizing the largest element.
        # Usually, the optimal solution uses the minimum number of elements required to cover 1..30, 
        # which is 9. Using more elements might allow a smaller D?
        # If we use more elements, we have more differences, which is "wasteful" but might allow 
        # packing them into a smaller range? 
        # No, Sidon sets are sparse. Adding elements forces them further apart.
        # So the optimal solution likely has k=9.
        
        # Let's try k=9 first. If no solution for D, try k=10, etc?
        # Or just let the solver decide the size?
        # The CSP solver naturally handles variable size if we don't fix it.
        # But we need to know when to stop adding elements.
        # We can stop adding elements if all targets are covered.
        
        def dfs(S, used_diffs, covered_mask, idx):
            if time.time() > deadline:
                return None
            
            # Check if all targets covered
            if (covered_mask & target_mask) == target_mask:
                return list(S)
            
            # If we have used too many elements, stop
            if idx > 31: # Max 32 elements including 0
                return None
                
            cur_max = S[-1] if S else 0
            
            # Pruning: Can we cover the rest?
            missing = target_mask & ~covered_mask
            num_missing = bin(missing).count('1')
            
            # Max elements we can add
            max_new_elements = D - cur_max
            if max_new_elements <= 0:
                return None
            
            # Max new differences possible with k_new elements
            # Current size n = len(S)
            # New elements x1...xk
            # Diffs: xj - si (si in S) and xj - xi (xi in new, i<j)
            # Count: k * n + k*(k-1)/2
            n = len(S)
            # We need to cover num_missing specific values.
            # The number of *new* differences generated is k*n + k*(k-1)/2.
            # But they must be distinct and not in used_diffs.
            # Simple bound: potential new diffs >= num_missing
            # Also, we can't exceed D.
            
            # Heuristic: Try to add elements that cover many missing diffs.
            
            # Generate candidates for next element x > cur_max
            # x can be from cur_max + 1 to D
            cands = []
            for x in range(cur_max + 1, D + 1):
                ok = True
                new_covered = 0
                for s in S:
                    d = x - s
                    if d in used_diffs:
                        ok = False
                        break
                    if 1 <= d <= 30:
                        new_covered |= mask_for_diff(d)
                if not ok:
                    continue
                
                new_missing_covered = new_covered & missing
                count_covered = bin(new_missing_covered).count('1')
                
                # Pruning: If we pick x, can the remaining elements cover the rest?
                # Remaining missing: num_missing - count_covered
                # Remaining elements capacity:
                # This is complex. Let's just sort by coverage.
                
                cands.append((count_covered, x))
            
            # Sort by coverage (desc), then x (asc) to try smaller x first?
            # Smaller x leaves more room for larger x later?
            # Usually, packing tight is good for minimizing max.
            cands.sort(key=lambda t: (-t[0], t[1]))
            
            # Limit candidates to avoid explosion
            # If D is large, this loop is expensive.
            # Optimization: Only consider x that cover at least 1 missing diff?
            # If x covers 0 missing diffs, it's only useful if it enables future diffs?
            # In Sidon sets, every element contributes to diffs.
            # But if it doesn't cover a *target* diff, it might be "waste".
            # However, it might be necessary to satisfy Sidon constraints for future elements.
            # Let's keep top 50 candidates.
            
            if len(cands) > 50:
                cands = cands[:50]
            
            for _, x in cands:
                new_used = set(used_diffs)
                new_mask = covered_mask
                for s in S:
                    d = x - s
                    new_used.add(d)
                    if 1 <= d <= 30:
                        new_mask |= mask_for_diff(d)
                
                S.append(x)
                res = dfs(S, new_used, new_mask, idx + 1)
                if res is not None:
                    return res
                S.pop()
                
            return None
        
        return dfs([0], set(), 0, 1)

    # Search for minimum D
    # Lower bound:
    # We need 30 diffs.
    # Min D is at least 30.
    # But Sidon constraint forces D to be larger.
    # Let's try from 31 upwards.
    for D in range(31, 100):
        res = search(D)
        if res is not None:
            return res
            
    # Fallback: return a valid set even if not minimal
    # Construct a large valid set
    S = [0]
    used = set()
    for d in range(1, 31):
        # Find x such that x - s = d for some s in S, or add new
        # Just add d * 100? No, must be Sidon.
        # Use a simple construction:
        # S = {0, 1, 4, 10, 18, 23, 29, 34, 41, 48, ...}
        # This is getting complicated.
        # Let's just return the first found in a wider search or a known solution.
        pass
    
    # If no solution found in 31..99, try a greedy construction
    S = [0]
    used = set()
    target = set(range(1, 31))
    while target:
        # Find smallest x > max(S) such that adding x keeps Sidon and covers at least one target
        best_x = None
        best_cover = -1
        for x in range(max(S) + 1, max(S) + 100):
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
            # Try further
            for x in range(max(S) + 100, max(S) + 500):
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
        
        if time.time() > deadline:
            break
            
    return S
