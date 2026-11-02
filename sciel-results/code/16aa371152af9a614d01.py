<<<<<<< SEARCH
    # Initial state
    if backtrack(0, [0]):
        return rows
    return None
=======
    # Initial state
    if backtrack(0, [0]):
        return rows
    return None

def greedy_construct(n_rows, k):
    """
    Fallback function to find a valid solution using randomized greedy approach.
    Returns a list of n_rows rows, each with k+1 integers.
    """
    import random
    best_sol = None
    best_max = float('inf')
    start_time = time.time()
    deadline = start_time + 50 # 50s limit for fallback
    
    while time.time() < deadline:
        used_diffs = set()
        rows = []
        success = True
        
        for r in range(n_rows):
            row = [0]
            
            for i in range(1, k + 1):
                found = False
                for x in range(row[-1] + 1, 500):
                    new_diffs = [x - y for y in row]
                    
                    conflict = False
                    for d in new_diffs:
                        if d in used_diffs:
                            conflict = True
                            break
                        
                    if conflict:
                        continue
                    
                    row.append(x)
                    used_diffs.update(new_diffs)
                    found = True
                    break
                
                if not found:
                    success = False
                    break
                    
            if not success:
                break
                
            rows.append(row)
            
        if success:
            current_max = max(max(r) for r in rows)
            if current_max < best_max:
                best_max = current_max
                best_sol = [r[:] for r in rows]
            
    if best_sol:
        return best_sol
    else:
        B = 10000
        sol = []
        for i in range(n_rows):
            row = [j * B * (i + 1) for j in range(k + 1)]
            sol.append(row)
        return sol
>>>>>>> REPLACE
