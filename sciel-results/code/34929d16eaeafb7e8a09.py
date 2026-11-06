import time

def _is_sidon(a):
    """Check if sorted list a is a Sidon set."""
    n = len(a)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _covers_1_to_30(diffs):
    for d in range(1, 31):
        if d not in diffs:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_set = None
    best_diam = float('inf')
    
    # Phase 1: Try known constructions
    known_sets = [
        [0, 1, 4, 10, 12, 28, 32, 45, 63],
    ]
    for candidate in known_sets:
        if time.time() > deadline:
            break
        if _is_sidon(candidate) and _covers_1_to_30(set(candidate[j] - candidate[i] for i in range(len(candidate)) for j in range(i+1, len(candidate)))):
            diam = candidate[-1]
            if diam < best_diam:
                best_diam = diam
                best_set = list(candidate)
    
    # Phase 2: Greedy construction
    def greedy_construct(seed=0):
        a = [0]
        used_diffs = set()
        covered = set()
        
        for _ in range(30):
            if time.time() > deadline:
                return None, float('inf')
            best_x = None
            best_score = -1
            best_new_covered = set()
            
            for x in range(1, 100):
                if x in a:
                    continue
                new_diffs = set()
                ok = True
                for v in a:
                    d = x - v
                    if d <= 0:
                        continue
                    if d in used_diffs or d in new_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                
                new_covered = new_diffs & set(range(1, 31)) - covered
                score = len(new_covered) * 100 - x
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_new_covered = new_covered
            
            if best_x is None:
                break
            
            a.append(best_x)
            a.sort()
            used_diffs |= best_new_covered
            covered |= best_new_covered
            
            if _covers_1_to_30(used_diffs):
                break
        
        if _covers_1_to_30(used_diffs) and len(a) <= 32:
            return a, a[-1]
        return None, float('inf')
    
    for seed in range(10):
        if time.time() > deadline - 30:
            break
        gset, gdiam = greedy_construct(seed)
        if gset is not None and gdiam < best_diam:
            best_set = gset
            best_diam = gdiam
    
    # Phase 3: Backtracking search with bitsets
    target_mask = 0
    for i in range(1, 31):
        target_mask |= (1 << i)
    
    best_diam_bt = [best_diam]
    best_set_bt = [best_set]
    
    def backtrack(S, used_mask, covered_mask, current_max):
        if time.time() > deadline:
            return
        if current_max >= best_diam_bt[0]:
            return
        if (covered_mask & target_mask) == target_mask:
            if current_max < best_diam_bt[0]:
                best_diam_bt[0] = current_max
                best_set_bt[0] = list(S)
            return
        n = len(S)
        if n >= 32:
            return
        
        missing_count = 0
        for d in range(1, 31):
            if not (covered_mask & (1 << d)):
                missing_count += 1
        remaining = 32 - n
        if remaining * n < missing_count:
            return
        
        start = S[-1] + 1 if S else 1
        upper = min(best_diam_bt[0] - 1, 80)
        if upper < start:
            return
        
        for x in range(start, upper + 1):
            if time.time() > deadline:
                return
            ok = True
            new_mask = 0
            for v in S:
                d = x - v
                bit = 1 << d
                if (used_mask & bit) or (new_mask & bit):
                    ok = False
                    break
                new_mask |= bit
            if not ok:
                continue
            new_used = used_mask | new_mask
            new_covered = covered_mask | new_mask
            S.append(x)
            backtrack(S, new_used, new_covered, x)
            S.pop()
    
    if time.time() < deadline - 20:
        backtrack([0], 0, 0, 0)
        if best_set_bt[0] is not None and best_diam_bt[0] < best_diam:
            best_set = best_set_bt[0]
            best_diam = best_diam_bt[0]
    
    # Phase 4: Local search to improve
    def local_search(a, deadline):
        a = sorted(a)
        best = list(a)
        best_diam = a[-1]
        
        for _ in range(5000):
            if time.time() > deadline:
                break
            
            n = len(a)
            if n < 2:
                break
            
            found = False
            last_idx = n - 1
            old_val = a[last_idx]
            
            for new_val in range(1, old_val):
                if new_val in a:
                    continue
                trial = a[:last_idx] + [new_val]
                trial = sorted(trial)
                diffs = set(trial[j] - trial[i] for i in range(len(trial)) for j in range(i+1, len(trial)))
                if _is_sidon(trial) and _covers_1_to_30(diffs):
                    a = trial
                    best = list(a)
                    best_diam = a[-1]
                    found = True
                    break
            
            if not found:
                for i in range(n - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = a[i]
                    for new_val in range(1, old_val):
                        if new_val in a:
                            continue
                        trial = a[:i] + [new_val] + a[i+1:]
                        trial = sorted(trial)
                        diffs = set(trial[j] - trial[i] for i2 in range(len(trial)) for j2 in range(i2+1, len(trial)))
                        if _is_sidon(trial) and _covers_1_to_30(diffs):
                            diam = trial[-1]
                            if diam < best_diam:
                                a = trial
                                best = list(a)
                                best_diam = diam
                                found = True
                                break
                    if found:
                        break
        
        return best, best_diam
    
    if best_set is not None and time.time() < deadline - 5:
        improved, improved_diam = local_search(best_set, deadline - 5)
        if improved_diam < best_diam:
            best_set = improved
            best_diam = improved_diam
    
    # Fallback: simple greedy
    if best_set is None:
        a = [0]
        used = set()
        for d in range(1, 31):
            if d in used:
                continue
            found = False
            for x in range(1, 200):
                if x in a:
                    continue
                new_diffs = set()
                ok = True
                for v in a:
                    diff = abs(x - v)
                    if diff in used or diff in new_diffs:
                        ok = False
                        break
                    new_diffs.add(diff)
                if ok:
                    a.append(x)
                    a.sort()
                    used.update(new_diffs)
                    found = True
                    break
            if not found:
                x = max(a) + d + 1
                while True:
                    new_diffs = set()
                    ok = True
                    for v in a:
                        diff = x - v
                        if diff in used or diff in new_diffs:
                            ok = False
                            break
                        new_diffs.add(diff)
                    if ok:
                        a.append(x)
                        a.sort()
                        used.update(new_diffs)
                        break
                    x += 1
        best_set = a
        best_diam = max(a)
    
    return best_set
