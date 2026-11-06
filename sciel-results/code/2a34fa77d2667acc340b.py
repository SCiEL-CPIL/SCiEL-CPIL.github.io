import time

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    best_solution = None
    best_diameter = float('inf')
    
    def dfs(A, D, missing, n_target):
        nonlocal best_solution, best_diameter
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        if len(A) == n_target:
            if not missing:
                diam = max(A)
                if diam < best_diameter:
                    best_diameter = diam
                    best_solution = A[:]
            return
        
        # Pruning
        k = n_target - len(A)
        max_new_diffs = k * len(A) + k * (k - 1) // 2
        if len(missing) > max_new_diffs:
            return
            
        candidates = []
        seen = set()
        
        # 1. Covering candidates
        for d in sorted(missing):
            for a in A:
                x = a + d
                if x <= max(A):
                    continue
                if x in seen:
                    continue
                seen.add(x)
                
                valid = True
                new_diffs = []
                for y in A:
                    diff = x - y
                    if diff in D:
                        valid = False
                        break
                    new_diffs.append(diff)
                
                if valid:
                    candidates.append((x, new_diffs, True)) # True = covers missing
        
        # 2. Blind candidates (if no covering candidates or to explore more)
        # We only do this if we haven't found a solution yet or to improve?
        # Actually, blind steps are rarely optimal.
        # But if covering steps are blocked, we might need them.
        # Let's add a few blind steps if candidates is empty.
        if not candidates:
            max_a = max(A)
            for i in range(1, 11): # Try next 10 integers
                x = max_a + i
                if x in seen:
                    continue
                seen.add(x)
                
                valid = True
                new_diffs = []
                for y in A:
                    diff = x - y
                    if diff in D:
                        valid = False
                        break
                    new_diffs.append(diff)
                
                if valid:
                    candidates.append((x, new_diffs, False))
        
        if not candidates:
            return
            
        candidates.sort(key=lambda c: c[0])
        
        for x, new_diffs, covers in candidates:
            if x >= best_diameter:
                continue
                
            new_A = A + [x]
            new_D = D.union(new_diffs)
            new_missing = missing.copy()
            for d in new_diffs:
                if d in new_missing:
                    new_missing.remove(d)
                    
            dfs(new_A, new_D, new_missing, n_target)
            
            if time.time() > deadline:
                return

    initial_missing = set(range(1, 31))
    
    # Try n=9
    dfs([0], set(), initial_missing, 9)
    
    if best_solution is None:
        dfs([0], set(), initial_missing, 10)
        
    if best_solution is None:
        dfs([0], set(), initial_missing, 11)

    if best_solution is None:
        # Fallback
        return [0]
        
    return best_solution
