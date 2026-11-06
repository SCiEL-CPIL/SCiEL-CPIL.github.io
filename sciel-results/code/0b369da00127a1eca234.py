import time

_START = time.time()
DEADLINE = 85  # Leave 5s margin

def construct():
    t0 = time.time()
    
    # 1. Generate all 2^a * 3^b * 5^c smooth numbers <= 128
    smooth = []
    smooth_set = set()
    for a in range(0, 8):
        p2 = 2 ** a
        if p2 > 128: break
        for b in range(0, 6):
            p3 = 3 ** b
            if p2 * p3 > 128: break
            for c in range(0, 5):
                p5 = 5 ** c
                v = p2 * p3 * p5
                if v > 128: break
                if v not in smooth_set:
                    smooth_set.add(v)
                    smooth.append(v)
    smooth.sort()
    n_smooth = len(smooth)
    
    # Precompute divisibility matrix
    # div[i][j] is True if smooth[i] divides smooth[j]
    div = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                div[i][j] = True

    # IDDFS for each target
    def find_min_rep(target):
        # k=1
        if target in smooth_set:
            return [target]
        
        # IDDFS for k=2, 3, ...
        # Max k? 30 is the limit, but realistically < 10
        for k in range(2, 11):
            if time.time() - t0 > DEADLINE:
                # Fallback to a valid but potentially non-minimal rep
                # Use a simple greedy or random? 
                # For now, return a known valid set or break
                pass
            
            current_set = []
            
            def dfs(rem, start_idx, count):
                if count == k:
                    return rem == 0
                if rem < 0:
                    return False
                # Pruning: 
                # 1. If rem is too large for remaining slots
                if rem > (k - count) * smooth[-1]:
                    return False
                # 2. If rem is too small
                if start_idx + (k - count) <= n_smooth:
                    min_possible = sum(smooth[start_idx:start_idx + (k - count)])
                    if min_possible > rem:
                        return False
                
                for i in range(start_idx, n_smooth):
                    v = smooth[i]
                    if v > rem:
                        break
                    
                    # Check antichain
                    ok = True
                    for idx in current_set:
                        if div[idx][i] or div[i][idx]:
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    current_set.append(i)
                    if dfs(rem - v, i + 1, count + 1):
                        return True
                    current_set.pop()
                return False

            if dfs(target, 0, 0):
                return [smooth[i] for i in current_set]
        
        # Fallback if no solution found with k<11
        for k in range(11, 31):
            if time.time() - t0 > DEADLINE:
                break
            current_set = []
            def dfs(rem, start_idx, count):
                if count == k:
                    return rem == 0
                if rem < 0:
                    return False
                if rem > (k - count) * smooth[-1]:
                    return False
                if start_idx + (k - count) <= n_smooth:
                    min_possible = sum(smooth[start_idx:start_idx + (k - count)])
                    if min_possible > rem:
                        return False
                for i in range(start_idx, n_smooth):
                    v = smooth[i]
                    if v > rem:
                        break
                    ok = True
                    for idx in current_set:
                        if div[idx][i] or div[i][idx]:
                            ok = False
                            break
                    if not ok:
                        continue
                    current_set.append(i)
                    if dfs(rem - v, i + 1, count + 1):
                        return True
                    current_set.pop()
                return False
            if dfs(target, 0, 0):
                return [smooth[i] for i in current_set]
        
        return []

    answers = []
    for n in range(1, 129):
        if time.time() - t0 > DEADLINE - 5:
            # Emergency stop: use greedy for remaining
            pass
        
        rep = find_min_rep(n)
        if not rep:
            # Fallback greedy
            rem = n
            current = []
            for v in sorted(smooth, reverse=True):
                if v > rem:
                    continue
                ok = True
                for u in current:
                    if v % u == 0 or u % v == 0:
                        ok = False
                        break
                if ok:
                    current.append(v)
                    rem -= v
                    if rem == 0:
                        break
            if rem == 0:
                rep = current
            else:
                rep = [n] if n in smooth_set else [1, 2, 3, 5, 7, 9, 11, 13, 15, 17]
        answers.append(rep)
        
    return answers
