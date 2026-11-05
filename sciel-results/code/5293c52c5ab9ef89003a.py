import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    full_mask = (1 << 30) - 1
    max_size = 32
    
    def dfs(S, used_diffs, covered_mask, target_size, D):
        if time.time() > deadline:
            return None
            
        if (covered_mask & full_mask) == full_mask:
            return list(S)
            
        if len(S) >= target_size:
            return None
            
        missing = full_mask & ~covered_mask
        if missing == 0:
            return list(S)
            
        rem = target_size - len(S)
        if rem == 0:
            return None
            
        # Pruning: Max new differences
        max_new_diffs = rem * len(S) + rem * (rem - 1) // 2
        if bin(missing).count('1') > max_new_diffs:
            return None
            
        candidates = []
        last = S[-1]
        for x in range(last + 1, D + 1):
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
            candidates.append((coverage_count, x))
            
        if not candidates:
            return None
            
        candidates.sort(key=lambda t: (-t[0], t[1]))
        
        for _, x in candidates:
            new_used = set(used_diffs)
            new_mask = covered_mask
            for s in S:
                d = x - s
                new_used.add(d)
                if 1 <= d <= 30:
                    new_mask |= (1 << (d - 1))
                    
            S.append(x)
            res = dfs(S, new_used, new_mask, target_size, D)
            if res is not None:
                return res
            S.pop()
            
        return None

    # Search for minimum D
    for D in range(30, 120):
        if time.time() > deadline - 5:
            break
            
        # Determine possible sizes n
        # n must be at least 9.
        # n must satisfy binom(n,2) <= D.
        max_n = 1
        while (max_n + 1) * max_n // 2 <= D:
            max_n += 1
        
        # We try n from 9 to max_n (capped at 32)
        # Smaller n is generally easier to satisfy Sidon
        for n in range(9, min(33, max_n + 1)):
            if time.time() > deadline - 2:
                break
                
            res = dfs([0], set(), 0, n, D)
            if res is not None:
                return res
                
    # Fallback greedy
    S = [0]
    used = set()
    target = set(range(1, 31))
    while target and len(S) < max_size:
        best_x = None
        best_score = -1
        for x in range(max(S) + 1, 500):
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
        if best_x is None:
            break
        S.append(best_x)
        for s in S[:-1]:
            used.add(best_x - s)
        target -= used
        if time.time() > deadline - 1:
            break
            
    return S
