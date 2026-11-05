import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Greedy Fallback
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    greedy_sol = greedy_construct()
    best_sol = greedy_sol
    best_scope = max(max(r) for r in greedy_sol) if greedy_sol else 0
    
    # If k is large, exact search is infeasible
    if k > 10:
        return greedy_sol
        
    # 2. Exact Search
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    lower_bound = total_diffs
    
    # Generate all Golomb rulers of length k with scope up to max_scope_limit
    # We set limit to be a bit higher than best_scope to ensure we find something better if possible
    # But we iterate M from lower_bound upwards.
    # We need the pool to contain all rulers with scope <= M.
    # So we generate up to a safe max, e.g., best_scope + 10 or 200.
    max_scope_limit = max(best_scope + 20, lower_bound + 20, 150)
    
    candidates = [] # list of (scope, mask, row_tuple)
    
    def generate_rulers(limit):
        rulers = []
        
        def backtrack(row, used_diffs):
            if time.time() > deadline:
                return
            if len(row) == k + 1:
                scope = row[-1]
                if scope <= limit:
                    mask = 0
                    for d in used_diffs:
                        mask |= (1 << d)
                    rulers.append((scope, mask, tuple(row)))
                return
            
            last = row[-1]
            # Next mark x
            # We can bound x. 
            # The remaining marks must be placed.
            # Simple bound: x <= limit
            
            for x in range(last + 1, limit + 1):
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    row.append(x)
                    backtrack(row, used_diffs | new_diffs)
                    row.pop()
                    
        backtrack([0], set())
        return rulers

    all_rulers = generate_rulers(max_scope_limit)
    all_rulers.sort(key=lambda x: x[0])
    
    # Search for M
    for M in range(lower_bound, max_scope_limit + 1):
        if time.time() > deadline:
            break
            
        # Filter candidates with scope <= M
        # Optimization: since all_rulers is sorted, we can find the cutoff index
        # But simple filter is fine for small lists.
        pool = [r for r in all_rulers if r[0] <= M]
        
        if len(pool) < n_rows:
            continue
            
        found = [None]
        
        def dfs(idx, used_mask, selected):
            if time.time() > deadline:
                return False
            
            if len(selected) == n_rows:
                found[0] = [list(r[2]) for r in selected]
                return True
            
            # Pruning
            if len(pool) - idx < n_rows - len(selected):
                return False
            
            for i in range(idx, len(pool)):
                scope, mask, row = pool[i]
                if mask & used_mask:
                    continue
                
                selected.append((scope, mask, row))
                if dfs(i + 1, used_mask | mask, selected):
                    return True
                selected.pop()
                
            return False
        
        if dfs(0, 0, []):
            sol = found[0]
            scope = max(max(r) for r in sol)
            if scope < best_scope:
                best_sol = sol
                best_scope = scope
            # Since we iterate M upwards, the first valid solution is the best for this M.
            # And since M is minimal, this is the global best.
            return sol
            
    return best_sol
