import time
def construct():
    deadline = time.time() + 115.0
    
    # 1. Greedy construction to find an initial valid solution and upper bound
    def greedy_construct():
        S = [0]
        used = set()
        target = set(range(1, 31))
        while target and len(S) < 32:
            best_x = None
            best_score = -1
            # Search up to a reasonable limit, say 200
            for x in range(max(S) + 1, 200):
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
                    # Prefer covering more targets, then smaller x
                    score = cover * 1000 - x
                    if score > best_score:
                        best_score = score
                        best_x = x
            if best_x is None:
                # Fallback: any valid x
                for x in range(max(S) + 1, 200):
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
        if not target:
            return S
        return None

    S_greedy = greedy_construct()
    D_max = max(S_greedy) if S_greedy else 100
    
    # 2. DFS to find minimum D
    full_mask = (1 << 30) - 1
    max_size = 32
    
    def dfs(S, used_diffs, covered_mask, D):
        if time.time() > deadline:
            return None
        
        if (covered_mask & full_mask) == full_mask:
            return list(S)
            
        missing = full_mask & ~covered_mask
        if missing == 0:
            return list(S)
            
        if len(S) >= max_size:
            return None
            
        remaining_slots = max_size - len(S)
        max_new_diffs = remaining_slots * len(S) + remaining_slots * (remaining_slots - 1) // 2
        if bin(missing).count('1') > max_new_diffs:
            return None
            
        candidates = []
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
            candidates.append((coverage_count, x))
            
        if not candidates:
            return None
            
        # Heuristic: Prefer higher coverage, then smaller x
        candidates.sort(key=lambda t: (-t[0], t[1]))
        
        # Limit candidates to avoid explosion
        limit = 50 if D <= 50 else 100
        for _, x in candidates[:limit]:
            new_used = set(used_diffs)
            new_mask = covered_mask
            for s in S:
                d = x - s
                new_used.add(d)
                if 1 <= d <= 30:
                    new_mask |= (1 << (d - 1))
                    
            S.append(x)
            res = dfs(S, new_used, new_mask, D)
            if res is not None:
                return res
            S.pop()
            
        return None

    # Search D from 30 to D_max
    for D in range(30, D_max + 1):
        if time.time() > deadline - 5:
            break
        res = dfs([0], set(), 0, D)
        if res is not None:
            return res
            
    if S_greedy:
        return S_greedy
    return [0]
