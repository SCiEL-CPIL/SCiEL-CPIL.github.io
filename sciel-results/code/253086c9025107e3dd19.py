import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline to be safe
    
    # We know size must be at least 9.
    # Let's iterate M from 1 upwards.
    # A perfect difference set of size 9 has diameter around 20-30?
    # Actually, for k=9, min diameter is known to be around 20?
    # Let's just search.
    
    for M in range(1, 1000):
        if time.time() > deadline:
            break
            
        result = _search(M)
        if result is not None:
            return result
            
    # Fallback: return a valid set if we found one in previous iterations?
    # No, we return the first one found.
    # If we hit deadline, we might not have found one.
    # But M is small, it should be fast.
    # If not, we can return a greedy result as fallback.
    return _greedy_fallback()

def _search(M):
    # DFS with backtracking
    S = [0]
    D = set()
    missing = set(range(1, 31))
    
    def dfs(S, D, missing, current_max):
        if not missing:
            return S[:]
        
        if current_max == M:
            return None
            
        # Pruning: check if enough capacity
        # Max diffs from adding all remaining (M - current_max) elements
        # Let k = len(S)
        # New elements j can be up to M - current_max
        # Max new diffs = j*k + j*(j-1)/2
        # We need at least len(missing) new diffs that are in 1..30?
        # Actually, we just need to cover missing.
        # The diffs generated must be distinct and not in D.
        # A loose bound: total diffs in final set must be >= len(D) + len(missing).
        # Final size K. K(K-1)/2 >= len(D) + len(missing).
        # This is weak.
        
        # Better: generate candidates
        candidates = []
        for x in range(current_max + 1, M + 1):
            new_diffs = [x - s for s in S]
            # Check collision with existing D
            if any(d in D for d in new_diffs):
                continue
            # Check collision within new_diffs (always distinct if s distinct)
            # Check if new_diffs introduces any diff > M? No, x<=M, s>=0.
            
            # Count how many missing diffs this x covers
            covers = 0
            for d in new_diffs:
                if d in missing:
                    covers += 1
            
            # Heuristic: prioritize x that covers more missing diffs
            candidates.append((covers, x, new_diffs))
            
        # Sort by covers descending, then x ascending
        candidates.sort(key=lambda t: (-t[0], t[1]))
        
        for covers, x, new_diffs in candidates:
            # If covers is 0, it might still be useful to bridge for future, 
            # but usually we want to cover missing.
            # However, sometimes we need to add an element that doesn't cover missing 
            # to enable other elements? 
            # In Sidon sets, structure is rigid.
            # Let's try all candidates, but sorted.
            
            new_D = D.union(new_diffs)
            new_missing = missing.difference(new_diffs)
            
            res = dfs(S + [x], new_D, new_missing, x)
            if res is not None:
                return res
                
        return None

    return dfs(S, D, missing, 0)

def _greedy_fallback():
    # Same as original but slightly improved
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                break
        else:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a
