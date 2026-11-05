import time

def construct():
    deadline = time.time() + 140.0
    TARGET = (1 << 30) - 1  # bits 0..29 represent diffs 1..30

    def popcount(x):
        return bin(x).count('1')

    def search(D):
        """Find Sidon set in [0,D] containing 0 covering all diffs 1..30.
        Returns the set as list, or None."""
        best = [None]
        
        # used_mask: bit d is 1 if difference d is used.
        # covered: bit d-1 is 1 if difference d (1..30) is covered.
        # S: list of elements.
        
        def dfs(S, used_mask, covered):
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
            
            # Generate candidates
            # We want to try x that cover many new diffs.
            # But we must check Sidon.
            
            cands = []
            for x in range(cur_max + 1, D + 1):
                # Check Sidon
                # new_diffs_mask will have bits set for x-s
                new_diffs_mask = 0
                new_cov = 0
                valid = True
                for s in S:
                    d = x - s
                    if d > D: # Should not happen if x<=D and s>=0
                        valid = False
                        break
                    bit = 1 << d
                    if (used_mask & bit):
                        valid = False
                        break
                    new_diffs_mask |= bit
                    if 1 <= d <= 30:
                        new_cov |= (1 << (d - 1))
                
                if not valid:
                    continue
                
                # Check if new_diffs_mask has internal collisions?
                # x-s are distinct for distinct s. So no internal collision.
                
                new_cov &= ~covered
                cnt = popcount(new_cov)
                cands.append((cnt, x, new_cov, new_diffs_mask))
            
            # Sort candidates: High coverage first, then smaller x (to keep diameter tight? 
            # Actually, if we are fixed D, smaller x leaves more room for larger x later?
            # No, we just need to fill gaps. Smaller x might be better to allow larger jumps later?
            # Usually, greedy by coverage is good.)
            cands.sort(key=lambda t: (-t[0], t[1]))
            
            # Heuristic limit? 
            # If we don't limit, and D is small, it's fast.
            # If D is large, it might be slow.
            # But we are iterating D from small to large.
            # Let's try without strict limit first, but maybe cap at 50 if D is large?
            # For now, let's process all.
            
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
    # Lower bound: We need 30 diffs.
    # Min diameter for 30 diffs in Sidon set?
    # Start from 30.
    for D in range(30, 100):
        if time.time() > deadline - 5:
            break
        res = search(D)
        if res is not None:
            return res
    
    # Fallback: Return a known valid set or greedy
    # A known Sidon set with small diameter?
    # If search fails, return a greedy result to be safe.
    S = [0]
    used = set()
    target = set(range(1, 31))
    while target and time.time() < deadline - 2:
        best_x = None
        best_cover = -1
        for x in range(max(S) + 1, max(S) + 500):
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
