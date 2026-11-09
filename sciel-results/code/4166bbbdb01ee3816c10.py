import time

L = 5354228880  # lcm(1..24)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    weights = [0] * 25
    for i in range(1, 25):
        weights[i] = L // i
    
    elements = list(range(24, 0, -1))
    n = len(elements)
    
    # Precompute max weight from index i to end (for pruning)
    max_future = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        max_future[i] = max(weights[elements[j]] for j in range(i, n))
    
    best_set = set()
    best_size = 0
    
    def backtrack(idx, selected, sums):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if len(selected) + (n - idx) <= best_size:
            return
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
        
        if idx >= n:
            return
        
        x = elements[idx]
        w = weights[x]
        
        # Try include x if no conflict
        if w not in sums:
            new_selected = selected + [x]
            max_w = max_future[idx + 1] if idx + 1 < n else 0
            new_sums = set()
            for s in sums:
                if s <= max_w:
                    new_sums.add(s)
                ns = s + w
                if ns <= max_w:
                    new_sums.add(ns)
            if w <= max_w:
                new_sums.add(w)
            if not new_sums:
                new_sums = {0}
            backtrack(idx + 1, new_selected, new_sums)
        
        # Try exclude x
        backtrack(idx + 1, selected, sums)
    
    backtrack(0, [], {0})
    
    # Local improvement
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for y in range(24, 0, -1):
                if y in best_set:
                    continue
                wy = weights[y]
                cur_w = [weights[z] for z in best_set]
                
                sums = {0}
                for w in cur_w:
                    new_s = set()
                    for s in sums:
                        ns = s + w
                        if ns <= wy:
                            new_s.add(ns)
                    sums.update(new_s)
                
                if wy in sums:
                    continue
                
                valid = True
                for z in best_set:
                    wz = weights[z]
                    others = [weights[zz] for zz in best_set if zz != z] + [wy]
                    sz = {0}
                    for w in others:
                        if w > wz:
                            continue
                        new_sz = set()
                        for s in sz:
                            ns = s + w
                            if ns <= wz:
                                new_sz.add(ns)
                        sz.update(new_sz)
                    if wz in sz:
                        valid = False
                        break
                
                if valid:
                    best_set.add(y)
                    improved = True
                    break
    
    if not best_set:
        best_set = {24}
    
    return list(best_set)
