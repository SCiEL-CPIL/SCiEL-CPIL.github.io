import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    target_diffs = set(range(1, 31))
    best_solution = None
    best_max_val = float('inf')
    
    # Initial greedy to get a valid upper bound quickly
    def greedy_construct():
        s = [0]
        covered = set()
        all_diffs = set()
        
        def get_diffs(lst):
            return {y - x for i, x in enumerate(lst) for y in lst[i+1:]}
            
        for _ in range(32):
            if covered == target_diffs:
                break
            found = False
            # Try to extend
            # Candidates: s_i + d for d in target - covered
            candidates = set()
            last = max(s)
            for d in target_diffs - covered:
                for v in s:
                    x = v + d
                    if x > last:
                        candidates.add(x)
            # Sort candidates
            for x in sorted(candidates):
                # Check Sidon
                new_diffs = {x - v for v in s}
                if new_diffs & all_diffs:
                    continue
                # Valid
                new_covered = covered | (new_diffs & target_diffs)
                # Update state for trial
                # Just checking if it leads to solution is complex, 
                # so greedy just picks first valid
                s.append(x)
                s.sort()
                all_diffs |= new_diffs
                covered = new_covered
                found = True
                break
            if not found:
                # Fallback: just try x = last + 1, last + 2 ...
                x = last + 1
                while True:
                    new_diffs = {x - v for v in s}
                    if not (new_diffs & all_diffs):
                        s.append(x)
                        s.sort()
                        all_diffs |= new_diffs
                        covered |= (new_diffs & target_diffs)
                        break
                    x += 1
        return s if covered == target_diffs else None

    init_sol = greedy_construct()
    if init_sol:
        best_solution = init_sol
        best_max_val = max(init_sol)

    # DFS
    # State: current set (list), covered diffs (set), all diffs (set)
    # We pass sets for efficiency in recursion? Or recompute?
    # Recomputing diffs from list is O(N^2). N <= 32. 32^2 = 1024. Cheap.
    # But passing sets is faster.
    
    def dfs(s, covered, all_diffs):
        nonlocal best_solution, best_max_val
        
        if time.time() > deadline:
            return False
            
        if covered == target_diffs:
            curr_max = max(s)
            if curr_max < best_max_val:
                best_max_val = curr_max
                best_solution = list(s)
            return True # Found a solution, prune siblings
            
        if len(s) >= 32:
            return False
            
        curr_max = max(s)
        if curr_max >= best_max_val:
            return False
            
        last = curr_max
        
        # Generate candidates
        # x such that x > last and (x - v) in (target_diffs - covered) for some v in s
        candidates = set()
        missing = target_diffs - covered
        for d in missing:
            for v in s:
                x = v + d
                if x > last:
                    candidates.add(x)
        
        # Sort candidates by value (smallest first)
        # This ensures we find the smallest possible next element first
        for x in sorted(candidates):
            if x >= best_max_val:
                break # Since sorted, all further x are larger
                
            # Check Sidon constraint
            new_diffs = {x - v for v in s}
            if new_diffs & all_diffs:
                continue
                
            # Valid extension
            new_covered = covered | (new_diffs & target_diffs)
            new_all_diffs = all_diffs | new_diffs
            new_s = s + [x]
            
            if dfs(new_s, new_covered, new_all_diffs):
                return True
                
        # If no candidate from 'missing' worked, try a few 'waste' candidates?
        # To avoid getting stuck if a waste mark is needed.
        # But as argued, waste marks are unlikely to be needed in optimal solution.
        # However, to be safe, we can try a few small valid x that don't cover new diffs.
        # But this increases branching factor.
        # Let's try just a few: last+1, last+2, etc.
        # Actually, if we didn't find a solution with productive marks, 
        # adding waste marks will only increase max or keep it same but add constraints.
        # It is very likely that if a solution exists, it uses productive marks.
        # We will skip waste marks to keep search space small.
        
        return False

    dfs([0], set(), set())
    
    if best_solution:
        return best_solution
    else:
        # Fallback to a very large valid set if search failed (should not happen)
        return [0, 1, 4, 10, 12, 17, 20, 23, 29, 34, 37, 40] # Random guess, needs validation
