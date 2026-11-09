import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    req_diffs = set(range(1, 31))
    best_max = float('inf')
    best_set = None

    def check_add(current_set, used_diffs, x):
        new_diffs = []
        for a in current_set:
            d = abs(x - a)
            if d in used_diffs:
                return False, None
            new_diffs.append(d)
        if len(set(new_diffs)) != len(new_diffs):
            return False, None
        return True, new_diffs

    def dfs(current_set, used_diffs, req_diffs):
        nonlocal best_max, best_set
        
        if time.time() > deadline:
            return

        if not req_diffs:
            current_max = max(current_set)
            if current_max < best_max:
                best_max = current_max
                best_set = list(current_set)
            return

        if len(current_set) >= 32:
            return
            
        current_max = max(current_set)
        if current_max >= best_max:
            return

        candidates = set()
        for a in current_set:
            for d in req_diffs:
                c = a + d
                if c > current_max and c < best_max:
                    candidates.add(c)
                
        scored_candidates = []
        for x in candidates:
            valid, new_diffs = check_add(current_set, used_diffs, x)
            if valid:
                covered = sum(1 for d in new_diffs if d in req_diffs)
                scored_candidates.append((covered, x, new_diffs))
        
        scored_candidates.sort(key=lambda t: (-t[0], t[1]))
        
        for covered, x, new_diffs in scored_candidates[:20]:
            new_used_diffs = used_diffs.union(new_diffs)
            new_req_diffs = req_diffs - set(new_diffs)
            dfs(current_set + [x], new_used_diffs, new_req_diffs)

    dfs([0], set(), set(range(1, 31)))
    
    if best_set is None:
        # Fallback
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            found = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a

    return best_set
