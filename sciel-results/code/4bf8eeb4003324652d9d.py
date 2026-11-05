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
        S = [0]
        used_diffs = set()
        covered_mask = 0
        
        def dfs(S, used_diffs, covered_mask):
            if time.time() > deadline:
                return None
            
            # Check if all targets covered
            if (covered_mask & target_mask) == target_mask:
                return list(S)
            
            # Pruning: Can we cover the rest?
            missing = target_mask & ~covered_mask
            num_missing = bin(missing).count('1')
            
            cur_max = S[-1]
            # Max elements we can add
            max_new_elements = D - cur_max
            if max_new_elements <= 0:
                return None
            
            # Max new differences possible
            # New elements x1...xk.
            # Diffs: xj - si (si in S) and xj - xi (xi in new)
            # Count: k * len(S) + k*(k-1)/2
            # This is an upper bound on *distinct* new differences.
            # Actually, we just need to cover `num_missing` specific values.
            # A tighter bound:
            # Each new element x can cover at most len(S) + (index of x in new) differences?
            # No, x - s for s in S are distinct if S is Sidon.
            # So each new element covers at most len(S) new diffs.
            # Plus diffs between new elements.
            # Total potential new diffs = max_new_elements * len(S) + max_new_elements * (max_new_elements - 1) // 2
            potential = max_new_elements * len(S) + max_new_elements * (max_new_elements - 1) // 2
            if potential < num_missing:
                return None
            
            # Generate candidates
            # We want to try x values that cover many missing diffs.
            cands = []
            for x in range(cur_max + 1, D + 1):
                # Check Sidon
                ok = True
                new_covered = 0
                for s in S:
                    d = x - s
                    if d in used_diffs:
                        ok = False
                        break
                    if d in target_diffs:
                        new_covered |= mask_for_diff(d)
                if not ok:
                    continue
                # Count how many *new* missing diffs this covers
                new_missing_covered = new_covered & missing
                cands.append((bin(new_missing_covered).count('1'), x))
            
            # Sort by coverage (desc), then x (asc)
            cands.sort(key=lambda t: (-t[0], t[1]))
            
            # Limit candidates to speed up
            # If there are too many, pick the best 50
            if len(cands) > 50:
                cands = cands[:50]
            
            for _, x in cands:
                # Re-check Sidon and update state
                # We already checked Sidon in candidate gen, but let's be safe or just trust it?
                # Trusting it is faster.
                
                new_used = set(used_diffs)
                new_mask = covered_mask
                for s in S:
                    d = x - s
                    new_used.add(d)
                    if 1 <= d <= 30:
                        new_mask |= mask_for_diff(d)
                
                S.append(x)
                res = dfs(S, new_used, new_mask)
                if res is not None:
                    return res
                S.pop()
                
            return None
        
        return dfs(S, used_diffs, covered_mask)

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
