import time

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def greedy():
        a = [0]
        used = set()
        for d in range(1, 31):
            if d in used:
                continue
            found = False
            for x in range(max(a) + 1, max(a) + 1000):
                new_diffs = [x - v for v in a]
                if all(d not in used for d in new_diffs) and len(set(new_diffs)) == len(new_diffs):
                    a.append(x)
                    used.update(new_diffs)
                    found = True
                    break
            if not found:
                # Fallback: add large element
                x = 2 * max(a) + 100
                while True:
                    new_diffs = [x - v for v in a]
                    if all(d not in used for d in new_diffs) and len(set(new_diffs)) == len(new_diffs):
                        a.append(x)
                        used.update(new_diffs)
                        break
                    x += 1
        return a

    init_sol = greedy()
    best_diameter = max(init_sol)
    best_solution = init_sol
    
    # DFS
    def dfs(S, all_diffs, missing_mask, max_elem):
        nonlocal best_diameter, best_solution
        
        if time.time() > deadline:
            return
            
        if missing_mask == 0:
            if max_elem < best_diameter:
                best_diameter = max_elem
                best_solution = list(S)
            return
            
        if len(S) > 31:
            return
            
        if max_elem >= best_diameter:
            return
            
        # Iterate next element x
        # Start from max_elem + 1
        # Stop at best_diameter - 1
        for x in range(max_elem + 1, best_diameter):
            # Check validity
            valid = True
            new_diffs = []
            for s in S:
                d = x - s
                if d in all_diffs:
                    valid = False
                    break
                new_diffs.append(d)
                
            if not valid:
                continue
                
            # Check new_diffs for internal duplicates (impossible if S unique)
            # Update missing_mask
            new_missing_mask = missing_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_missing_mask &= ~(1 << (d - 1))
                    
            # Heuristic: If we didn't reduce missing_mask, maybe skip?
            # No, we might need to build up to cover large diffs.
            
            # Recurse
            new_S = S + (x,)
            new_all_diffs = set(all_diffs)
            new_all_diffs.update(new_diffs)
            
            dfs(new_S, frozenset(new_all_diffs), new_missing_mask, x)
            
            # If we found a solution with smaller diameter, the loop bound `best_diameter` updates automatically.
            # But we are in a for loop with a fixed upper bound from the start of the loop?
            # No, `best_diameter` is nonlocal. The range object is created once.
            # So if we find a better solution, we won't prune the current loop's remaining iterations immediately.
            # We can add a check inside the loop: if x >= best_diameter: break
            
            if x >= best_diameter:
                break

    initial_S = (0,)
    initial_diffs = frozenset()
    initial_missing = (1 << 30) - 1
    
    dfs(initial_S, initial_diffs, initial_missing, 0)
    
    return best_solution

if __name__ == "__main__":
    res = construct()
    print(res)
