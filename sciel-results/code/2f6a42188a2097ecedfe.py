import time

def construct():
    deadline = time.time() + 115.0
    
    target_mask = (1 << 30) - 1
    
    def search(D, max_time):
        start_time = time.time()
        best_result = [None]
        
        used_diffs = [False] * (D + 1)
        
        def dfs(S, covered_mask, depth):
            if time.time() > max_time:
                return False
            
            if (covered_mask & target_mask) == target_mask:
                best_result[0] = list(S)
                return True
            
            cur_max = S[-1] if S else 0
            if cur_max >= D:
                return False
            
            missing = target_mask & ~covered_mask
            num_missing = bin(missing).count('1')
            
            max_add = min(D - cur_max, 32 - len(S))
            potential = max_add * len(S) + max_add * (max_add - 1) // 2
            if potential < num_missing:
                return False
            
            cands = []
            for x in range(cur_max + 1, D + 1):
                ok = True
                new_mask = 0
                for s in S:
                    d = x - s
                    if d <= D and used_diffs[d]:
                        ok = False
                        break
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))
                if not ok:
                    continue
                
                new_cover = new_mask & missing
                cnt = bin(new_cover).count('1')
                if cnt > 0:
                    cands.append((cnt, x, new_mask))
            
            cands.sort(key=lambda t: (-t[0], t[1]))
            
            for cnt, x, new_mask in cands[:50]:
                changes = []
                for s in S:
                    d = x - s
                    changes.append(d)
                    if d <= D:
                        used_diffs[d] = True
                
                S.append(x)
                dfs(S, covered_mask | new_mask, depth + 1)
                S.pop()
                
                for d in changes:
                    if d <= D:
                        used_diffs[d] = False
                
                if best_result[0] is not None:
                    return True
                if time.time() > max_time:
                    return False
        
        dfs([0], 0, 0)
        return best_result[0]

    for D in range(30, 100):
        if time.time() > deadline:
            break
        time_left = deadline - time.time()
        time_per_D = min(5.0, time_left * 0.8)
        if time_per_D < 0.5: break
        res = search(D, time.time() + time_per_D)
        if res is not None:
            return res
    
    S = [0]
    used = set()
    target = set(range(1, 31))
    while target and time.time() < deadline:
        best_x = None
        best_cover = -1
        for x in range(max(S) + 1, max(S) + 200):
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
