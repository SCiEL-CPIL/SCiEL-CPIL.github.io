"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import sys
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of length k+1,
    starting at 0, strictly increasing, such that all positive differences
    within each row are distinct across all rows.
    
    Approach: Randomized restart backtracking with constraint propagation.
    For each restart, try to build rows with a random ordering of candidate values,
    using forward checking to prune infeasible branches. Use a time budget per restart.
    """
    
    def total_diffs(n_rows, k):
        return n_rows * k * (k + 1) // 2
    
    def min_scope_bound(n_rows, k):
        return total_diffs(n_rows, k)
    
    def try_construct_backtrack(scope, n_rows, k, time_limit=5.0, seed=0):
        """
        Try to construct n_rows valid rows with all entries ≤ scope using backtracking.
        Uses randomized restarts within the backtracking for diversity.
        """
        rng = random.Random(seed)
        start_time = time.time()
        
        used_diffs = set()
        rows = []
        
        def backtrack(row_idx, row):
            if time.time() - start_time > time_limit:
                return None
            
            if row_idx == n_rows:
                return rows[:]
            
            if row_idx < len(rows):
                row = rows[row_idx]
            else:
                row = [0]
                rows.append(row)
            
            if len(row) == k + 1:
                if row_idx + 1 < n_rows:
                    return backtrack(row_idx + 1, [0])
                else:
                    return rows[:]
            
            prev = row[-1]
            candidates = list(range(prev + 1, scope + 1))
            rng.shuffle(candidates)
            
            for val in candidates:
                new_diffs = {val - a for a in row}
                
                if new_diffs & used_diffs:
                    continue
                
                row.append(val)
                used_diffs.update(new_diffs)
                
                result = backtrack(row_idx, row)
                if result is not None:
                    return result
                
                row.pop()
                used_diffs.difference_update(new_diffs)
            
            return None
        
        return backtrack(0, [0])
    
    def greedy_construct():
        """Fallback greedy construction."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def local_improve(rows, max_iter=5000):
        """Improve scope via local element replacement."""
        if rows is None:
            return None, None
        
        current_scope = max(max(row) for row in rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        
        for _ in range(max_iter):
            improved = False
            for row_idx in range(n_rows):
                row = rows[row_idx]
                for pos in range(len(row) - 1, 0, -1):
                    old_val = row[pos]
                    if old_val <= best_scope:
                        continue
                    
                    min_val = row[pos - 1] + 1
                    max_val = min(old_val - 1, best_scope)
                    
                    if max_val < min_val:
                        continue
                    
                    row_diffs = set()
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            row_diffs.add(row[j] - row[i])
                    
                    old_diffs = {old_val - a for a in row[:pos]}
                    
                    other_diffs = set()
                    for i, r in enumerate(rows):
                        if i == row_idx:
                            continue
                        for a in range(len(r)):
                            for b in range(a + 1, len(r)):
                                other_diffs.add(r[b] - r[a])
                    
                    for new_val in range(max_val, min_val - 1, -1):
                        new_diffs = {new_val - a for a in row[:pos]}
                        remaining = (row_diffs - old_diffs) | new_diffs
                        if remaining & other_diffs:
                            continue
                        
                        row[pos] = new_val
                        improved = True
                        break
                    
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        current_scope = max(max(row) for row in rows)
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [r[:] for r in rows]
        
        return best_rows, best_scope
    
    lower_bound = min_scope_bound(n_rows, k)
    best_rows = None
    best_scope = None
    
    # Try backtracking with increasing scopes and multiple restarts
    for scope in range(lower_bound, lower_bound + 50):
        for seed in range(10):
            result = try_construct_backtrack(scope, n_rows, k, time_limit=3.0, seed=seed)
            if result is not None:
                best_rows = result
                best_scope = scope
                improved_rows, improved_scope = local_improve(result, max_iter=5000)
                if improved_rows is not None and improved_scope < best_scope:
                    best_rows = improved_rows
                    best_scope = improved_scope
                break
        if best_rows is not None:
            break
    
    if best_rows is None:
        best_rows = greedy_construct()
        best_scope = max(max(row) for row in best_rows)
        improved_rows, improved_scope = local_improve(best_rows, max_iter=10000)
        if improved_rows is not None and improved_scope < best_scope:
            best_rows = improved_rows
            best_scope = improved_scope
    
    return best_rows
# EVOLVE-BLOCK-END
