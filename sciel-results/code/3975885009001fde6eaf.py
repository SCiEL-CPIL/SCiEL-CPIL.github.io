import time

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin for 150s limit
    
    best_solution = None
    best_diameter = float('inf')
    
    initial_set = [0]
    initial_used = set()
    initial_missing = set(range(1, 31))
    
    # To speed up, we can pre-compute or just use sets
    # Using a list for S is fine for small N
    
    def dfs(S, used, missing):
        nonlocal best_solution, best_diameter
        
        # Time check
        if time.time() - start_time > timeout:
            return
        
        # Success condition
        if not missing:
            diam = max(S)
            if diam < best_diameter:
                best_diameter = diam
                best_solution = list(S)
            return
        
        # Constraint: Max size
        if len(S) >= 32:
            return
            
        # Pruning: Current max is already >= best
        max_s = max(S)
        if max_s >= best_diameter:
            return
            
        # Generate candidates
        # Candidates must be such that adding x creates at least one difference in 'missing'
        candidates = set()
        # Limit x to best_diameter - 1 to ensure improvement potential
        limit = int(best_diameter) if best_diameter < float('inf') else 1000
        
        for y in S:
            for d in missing:
                x1 = y + d
                if 0 <= x1 < limit:
                    candidates.add(x1)
                x2 = y - d
                if 0 <= x2 < limit:
                    candidates.add(x2)
        
        # If no candidates, we are stuck (cannot cover any missing diff with valid x < limit)
        if not candidates:
            return
            
        # Sort candidates to explore smaller values first
        for x in sorted(candidates):
            if x in S:
                continue
                
            # Check Sidon property
            valid = True
            new_diffs = []
            for y in S:
                d = abs(x - y)
                if d in used:
                    valid = False
                    break
                new_diffs.append(d)
                
            if not valid:
                continue
                
            # Update state
            new_used = used.union(new_diffs)
            new_missing = set(missing)
            for d in new_diffs:
                if d in new_missing:
                    new_missing.remove(d)
                    
            S_new = S + [x]
            S_new.sort()
            
            dfs(S_new, new_used, new_missing)
            
            # Backtrack is handled by creating new objects

    dfs(initial_set, initial_used, initial_missing)
    
    if best_solution is None:
        # Fallback to the seed's greedy algorithm to ensure a valid return
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
        
    return best_solution
