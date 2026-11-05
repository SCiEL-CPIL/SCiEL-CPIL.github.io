import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    TARGET = (1 << 30) - 1  # bits 0..29 represent diffs 1..30

    def popcount(x):
        return bin(x).count('1')

    def search(D):
        """Find Sidon set in [0,D] containing 0 covering all diffs 1..30.
        Returns the set as list, or None."""
        best = [None]
        
        def dfs(S, used_mask, covered):
            # Time check
            if time.time() > deadline - 2.0:
                return False
            
            if (covered & TARGET) == TARGET:
                if best[0] is None:
                    best[0] = list(S)
                return True
            
            n = len(S)
            if n >= 32:
                return False
            
            cur_max = S[-1] if S else 0
            if cur_max >= D:
                return False
            
            missing = TARGET & ~covered
            nmiss = popcount(missing)
            
            # Pruning: Max possible new diffs
            max_rem = min(D - cur_max, 32 - n)
            if max_rem <= 0:
                return False
            
            # Each new element x adds |S| diffs.
            # The k new elements add k*|S| + k*(k-1)/2 diffs total.
            max_new_diffs = max_rem * n + max_rem * (max_rem - 1) // 2
            if max_new_diffs < nmiss:
                return False
            
            # Additional Pruning:
            # The largest missing difference must be coverable.
            # Max possible difference is D - 0 = D (if 0 is in S).
            # Actually, max diff is D - min(S). Since 0 is in S, max diff is D.
            # So if max(missing) > D, prune.
            # But missing is a bitmask of 1..30. So max missing is <= 30.
            # Since we start D from 30, this is always true initially.
            
            cands = []
            for x in range(cur_max + 1, D + 1):
                # Check Sidon
                new_diffs_mask = 0
                new_cov = 0
                valid = True
                for s in S:
                    d = x - s
                    bit = 1 << d
                    if (used_mask & bit):
                        valid = False
                        break
                    new_diffs_mask |= bit
                    if 1 <= d <= 30:
                        new_cov |= (1 << (d - 1))
                
                if not valid:
                    continue
                
                new_cov &= ~covered
                cnt = popcount(new_cov)
                cands.append((cnt, x, new_cov, new_diffs_mask))
            
            # Sort candidates: High coverage first, then smaller x
            cands.sort(key=lambda t: (-t[0], t[1]))
            
            for cnt, x, new_cov, new_diffs_mask in cands:
                S.append(x)
                if dfs(S, used_mask | new_diffs_mask, covered | new_cov):
                    S.pop()
                    return True
                S.pop()
            
            return False
        
        dfs([0], 0, 0)
        return best[0]

    # Try increasing diameters
    best_result = None
    for D in range(30, 100):
        if time.time() > deadline - 5:
            break
        res = search(D)
        if res is not None:
            return res
    
    # Fallback: Return a known valid set or greedy
    # Greedy construction to ensure we return *something* valid if search fails
    S = [0]
    used = set()
    target = set(range(1, 31))
    while target and time.time() < deadline - 2:
        best_x = None
        best_cover = -1
        # Search in a wider range for the greedy step
        for x in range(max(S) + 1, max(S) + 1000):
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
            break
        S.append(best_x)
        for s in S[:-1]:
            used.add(best_x - s)
        target -= used
    return S
