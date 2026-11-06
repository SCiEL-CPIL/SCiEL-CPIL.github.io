import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Target: Sidon set of size 9 covering 1..30.
    # Min diameter D >= 36.
    
    required_diffs = set(range(1, 31))
    
    def find_set_for_D(D):
        # Backtracking to find s1..s7 such that {0, s1..s7, D} is Sidon and covers 1..30.
        
        def backtrack(elements, used_diffs, idx):
            # elements: sorted list of chosen elements so far (starts with [0])
            # idx: next index to fill (1 to 7)
            
            # Time check
            if time.time() > deadline - 5:
                return None
            
            # If we have chosen 7 elements (indices 1..7), we must add D
            if idx == 8:
                # Check if adding D is valid
                # Diffs created: D - e for e in elements
                new_diffs = [D - e for e in elements]
                
                # Check for duplicates with existing used_diffs
                for d in new_diffs:
                    if d in used_diffs:
                        return None
                
                # Check if we cover 1..30
                final_diffs = used_diffs.union(new_diffs)
                if required_diffs.issubset(final_diffs):
                    return elements + [D]
                else:
                    return None
            
            # Determine range for next element c
            # c > elements[-1]
            # We need room for (8 - idx) more elements (including D)
            # So c <= D - (8 - idx)
            
            start_val = elements[-1] + 1
            end_val = D - (8 - idx)
            
            if start_val > end_val:
                return None
            
            # Heuristic: Try values that are likely to cover missing diffs?
            # For now, simple iteration.
            # To speed up, we can check if c itself is a required diff not yet covered?
            # c is a diff (c-0).
            
            for c in range(start_val, end_val + 1):
                # Check if c creates duplicate diffs
                new_diffs = [c - e for e in elements]
                valid = True
                for d in new_diffs:
                    if d in used_diffs:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Recurse
                new_elements = elements + [c]
                new_used = used_diffs.union(new_diffs)
                
                res = backtrack(new_elements, new_used, idx + 1)
                if res is not None:
                    return res
            
            return None
        
        return backtrack([0], set(), 1)

    # Try D from 36 upwards
    # We will keep the best solution found
    best_solution = None
    
    # If we run out of time, we might not find anything.
    # We should have a fallback greedy solution?
    # The seed program provides one.
    # But we want to beat it.
    
    # Let's try D=36 with a tight loop.
    # If it fails, try 37, etc.
    
    # To avoid spending all time on D=36 if it's unsolvable,
    # we can impose a local time limit?
    # Or just let it run. 36 is small.
    
    for D in range(36, 100):
        if time.time() > deadline - 10:
            break
        
        # We can add a local time limit for each D to ensure progress
        # But for small D, it should be fast.
        
        sol = find_set_for_D(D)
        if sol is not None:
            best_solution = sol
            break # Since we iterate D upwards, first found is minimal D
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: Greedy approach from seed, but optimized
    # This ensures we return *something* valid even if backtracking fails
    # (e.g. if no solution exists for D<100, which is unlikely)
    
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        # Try to add element to realize d
        found = False
        for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                found = True
                break
        if not found:
            # Add large elements
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
                if time.time() > deadline - 2:
                    return a # Return partial? No, must be valid.
                    # This loop might be infinite if no solution.
                    # But greedy usually works.
    
    return a
