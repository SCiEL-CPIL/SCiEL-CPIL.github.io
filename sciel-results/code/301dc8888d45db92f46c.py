# EVOLVE-BLOCK-START

def construct(n_rows, k):
    deadline = time.time() + 115
    
    if n_rows == 7 and k == 5:
        return _solve_7_5_optimized(deadline)
        
    # Fallback
    best_rows = None
    best_scope = float('inf')
    for seed in range(10):
        if time.time() >= deadline - 10:
            break
        rows = _greedy_construct(n_rows, k, seed)
        if rows:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    if best_rows:
        best_rows, best_scope = _simulated_annealing(best_rows, n_rows, k, deadline)
    return best_rows

def _solve_7_5_optimized(deadline):
    """
    Optimized solver for (7,5) using Backtracking with Greedy Candidates.
    """
    n_rows = 7
    k = 5
    rng = random.Random(12345)
    
    best_solution = None
    best_scope = float('inf')
    
    # We will try to find a solution with scope < best_scope.
    # Start with a high bound and decrease? Or just search for any solution and minimize?
    # Backtracking naturally finds a solution. We can prune if current max scope > best_scope.
    
    # Let's use a beam search / best-first search over partial solutions.
    # State: (rows_so_far, used_diffs, current_max_scope)
    # Priority: current_max_scope
    
    import heapq
    
    # Initial state: empty
    # We need to generate candidates for the first row.
    # To keep it fast, we limit the number of candidates per node.
    
    # We will use a simple DFS with pruning and randomization to avoid getting stuck.
    # Actually, let's use the "Greedy + Backtrack" idea.
    
    def _search(rows, used_diffs, current_max_scope):
        nonlocal best_solution, best_scope
        
        if time.time() > deadline - 5:
            return False
            
        if len(rows) == n_rows:
            if current_max_scope < best_scope:
                best_scope = current_max_scope
                best_solution = [list(r) for r in rows]
            return True
            
        # Generate candidates for the next row
        # 1. Greedy smallest
        # 2. Random variations
        
        candidates = []
        
        # Greedy candidate
        greedy_row = _greedy_next_row(used_diffs, k)
        if greedy_row:
            g_scope = max(greedy_row)
            if g_scope < best_scope: # Prune
                candidates.append(greedy_row)
                
        # Random candidates
        # Generate 10 random rows
        for _ in range(10):
            row = _random_valid_row(used_diffs, k, best_scope, rng)
            if row:
                r_scope = max(row)
                if r_scope < best_scope:
                    candidates.append(row)
                    
        # Sort candidates by scope
        candidates.sort(key=lambda r: max(r))
        
        # Try candidates
        for row in candidates:
            row_diffs = _compute_diffs(row)
            if row_diffs & used_diffs:
                continue
                
            new_max_scope = max(current_max_scope, max(row))
            
            if new_max_scope >= best_scope:
                continue
                
            rows.append(row)
            new_used = used_diffs | row_diffs
            
            if _search(rows, new_used, new_max_scope):
                # Found a better solution, but we might want to find even better?
                # For now, return True to stop this branch.
                # To find the absolute best, we should continue, but time is limited.
                rows.pop()
                return True
            else:
                rows.pop()
                
        return False
        
    # Initial call
    _search([], set(), 0)
    
    if best_solution:
        return best_solution
        
    # Fallback to greedy + SA if backtracking failed
    rows = _greedy_construct(7, 5, 0)
    if rows:
        rows, _ = _simulated_annealing(rows, 7, 5, deadline)
        return rows
        
    return None

def _greedy_next_row(used_diffs, k):
    """Find the smallest valid row given used_diffs."""
    row = [0]
    current_x = 0
    while len(row) <= k:
        current_x += 1
        new_diffs = {current_x - a for a in row}
        if not (new_diffs & used_diffs) and len(new_diffs) == len(row): # Internal distinctness check?
            # Wait, new_diffs size is len(row). If any duplicate, size < len(row).
            # Also need to check if new_diffs are distinct among themselves?
            # Yes, len(new_diffs) == len(row) ensures distinctness with previous elements.
            # But we also need to ensure the new element doesn't create duplicate diffs WITHIN the new set?
            # No, the set `new_diffs` contains diffs between current_x and all previous elements.
            # If they are distinct, it's fine.
            row.append(current_x)
        else:
            # If conflict, we can't just skip current_x and keep the same row?
            # The greedy algorithm in the original code did:
            # x += 1
            # new = {x - a for a in row}
            # if not new & used:
            #    row.append(x)
            # This assumes that if x is bad, x+1 might be good.
            # It does NOT backtrack. It just tries the next integer.
            # This is valid because we are constructing the row sequentially.
            pass
    return row if len(row) == k + 1 else None

def _random_valid_row(used_diffs, k, max_scope, rng):
    """Generate a random valid row disjoint from used_diffs."""
    for _ in range(50): # Attempts
        # Pick k distinct random numbers from 1..max_scope
        # Sort them
        nums = sorted(rng.sample(range(1, max_scope + 1), k))
        row = [0] + nums
        
        # Check internal distinctness
        diffs = set()
        valid = True
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in diffs:
                    valid = False
                    break
                diffs.add(d)
        if not valid:
            continue
            
        # Check global distinctness
        if diffs & used_diffs:
            continue
            
        return row
        
    return None

def _greedy_construct(n_rows, k, seed=0):
    """Greedy construction trying to minimize scope."""
    rng = random.Random(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used and len(new) == len(row):
                row.append(x)
                used |= new
        rows.append(row)
    
    return rows

def _scope(rows):
    return max(max(row) for row in rows)

def _compute_diffs(row):
    """Compute all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _simulated_annealing(rows, n_rows, k, deadline):
    """Simulated annealing to escape local minima."""
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    
    rng = random.Random(42)
    start_temp = 50.0
    end_temp = 0.1
    cooling_rate = 0.995
    
    row_diffs = [_compute_diffs(r) for r in current_rows]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
    
    temp = start_temp
    iterations = 0
    max_iterations = 50000
    
    while time.time() < deadline - 10 and temp > end_temp and iterations < max_iterations:
        iterations += 1
        
        ri = rng.randrange(n_rows)
        row = current_rows[ri]
        if len(row) <= 1:
            continue
        
        pos = rng.randrange(1, len(row))
        old_val = row[pos]
        
        delta = rng.randint(-15, 15)
        if delta == 0:
            continue
        new_val = old_val + delta
        
        if new_val <= 0:
            continue
        if pos > 0 and new_val <= row[pos - 1]:
            continue
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            continue
        
        old_diffs = _compute_diffs(row)
        
        row[pos] = new_val
        new_diffs = _compute_diffs(row)
        
        expected_diff_count = len(row) * (len(row) - 1) // 2
        if len(new_diffs) != expected_diff_count:
            row[pos] = old_val
            continue
        
        other_diffs = all_diffs - old_diffs
        if new_diffs & other_diffs:
            row[pos] = old_val
            continue
        
        new_scope = _scope(current_rows)
        delta_scope = new_scope - current_scope
        
        if delta_scope < 0 or rng.random() < math.exp(-delta_scope / temp):
            all_diffs = (all_diffs - old_diffs) | new_diffs
            row_diffs[ri] = new_diffs
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]
        else:
            row[pos] = old_val
        
        temp *= cooling_rate
    
    return best_rows, best_scope

# EVOLVE-BLOCK-END
