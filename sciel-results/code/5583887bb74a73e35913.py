"""Branch-and-bound with counting lower bounds for difference-triangle sets."""
import time
import random
import sys
from math import comb

# EVOLVE-BLOCK-START

def _valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs_set(row):
    n = len(row)
    s = set()
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _counting_lower_bound(n_rows_remaining, k, used_count, max_scope):
    """
    Lower bound on the number of distinct differences needed.
    Each row of size k+1 has C(k+1,2) = k*(k+1)/2 differences.
    Total diffs needed = n_rows_remaining * k*(k+1)//2
    Available diffs in [1..max_scope] = max_scope
    So we need max_scope >= used_count + n_rows_remaining * k*(k+1)//2
    This gives a lower bound on max_scope.
    """
    diffs_per_row = k * (k + 1) // 2
    total_needed = used_count + n_rows_remaining * diffs_per_row
    return total_needed

def _generate_row_candidates(k, max_scope, used_diffs, deadline, limit=2000):
    """
    Generate candidate rows (tuples of k+1 increasing ints starting at 0) 
    with all diffs distinct within the row and not in used_diffs.
    Returns list of (row_tuple, frozenset_of_diffs).
    Uses DFS with pruning.
    """
    results = []
    n = k + 1
    diffs_used_global = used_diffs  # frozenset or set
    chosen = [0]
    local_diffs = set()
    count = [0]
    
    def dfs(start, depth):
        if time.time() > deadline:
            return
        if count[0] >= limit:
            return
        if depth == n:
            row = tuple(chosen)
            diffs = frozenset(local_diffs)
            results.append((row, diffs))
            count[0] += 1
            return
        
        rem = n - depth
        # The next value v must satisfy:
        # 1. v >= start
        # 2. v <= max_scope - (rem - 1)  (leave room for remaining elements)
        # 3. For all a in chosen, v - a not in local_diffs and not in diffs_used_global
        # 4. v - a values must be distinct among themselves (automatically if v-a distinct for different a)
        
        max_next = min(max_scope - (rem - 1), max_scope)
        if max_next < start:
            return
            
        # Try values from start to max_next
        # Optimization: we can skip values that would create a diff already in use
        for v in range(start, max_next + 1):
            new_diffs = set()
            ok = True
            for a in chosen:
                d = v - a
                if d in local_diffs or d in diffs_used_global or d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                continue
            
            chosen.append(v)
            local_diffs.update(new_diffs)
            dfs(v + 1, depth + 1)
            chosen.pop()
            local_diffs.difference_update(new_diffs)
            
            if time.time() > deadline or count[0] >= limit:
                return
    
    dfs(1, 1)
    return results

def _bbs_search(n_rows, k, deadline, initial_bound=None):
    """
    Branch-and-bound search for difference triangle sets.
    Returns (rows, scope) or (None, inf).
    """
    best_rows = None
    best_scope = initial_bound if initial_bound is not None else float('inf')
    
    # Number of diffs per row
    diffs_per_row = k * (k + 1) // 2
    
    # Start with a reasonable upper bound from greedy
    def _quick_greedy(n_r, k_val, used, dl):
        """Fast greedy to get an initial feasible solution."""
        rows = []
        for _ in range(n_r):
            if time.time() > dl:
                return None
            row = [0]
            row_diffs = set()
            ok = True
            for _pos in range(k_val):
                found = False
                lo = row[-1] + 1
                for attempt in range(100):
                    if time.time() > dl:
                        return None
                    if attempt < 20:
                        v = lo + attempt
                    else:
                        v = lo + random.randint(0, 80)
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = v - a
                        if d in row_diffs or d in used or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        row.append(v)
                        row_diffs.update(new_diffs)
                        found = True
                        break
                if not found:
                    ok = False
                    break
            if ok:
                used.update(row_diffs)
                rows.append(row)
            else:
                return None
        return rows
    
    # Get initial bound via greedy
    used_init = set()
    greedy_rows = _quick_greedy(n_rows, k, used_init, deadline - 10)
    if greedy_rows is not None and _valid(greedy_rows, n_rows, k):
        best_scope = _max_entry(greedy_rows)
        best_rows = [r[:] for r in greedy_rows]
    
    if best_scope <= k:  # Can't do better than k in theory for small cases
        return best_rows, best_scope
    
    # B&B search
    # State: (row_idx, used_diffs, current_rows, current_max)
    # We'll do iterative deepening on scope
    
    # Try scopes from lower bound upward
    # Lower bound: need at least n_rows * diffs_per_row distinct diffs
    # Each diff is in [1..scope], so scope >= n_rows * diffs_per_row
    lb = n_rows * diffs_per_row
    
    # Also, for a single row of size k+1, the max entry must be at least k
    # And the differences must fit, so scope >= k
    
    # We'll try to find a solution with scope < best_scope
    # Search over scope values from max(lb, k) to best_scope - 1
    
    search_start = max(lb, k)
    if search_start >= best_scope:
        return best_rows, best_scope
    
    # For each candidate scope, try to find a valid assignment
    # Use backtracking with constraint propagation
    
    def _bbs_inner(scope, deadline):
        """
        Try to find n_rows rows with max entry <= scope.
        Returns list of rows or None.
        """
        # Generate all candidate rows for this scope
        # But generating all candidates for large scope is expensive.
        # Instead, we'll do on-the-fly generation with backtracking.
        
        rows = []
        used_diffs = set()
        
        def backtrack(row_idx, deadline):
            if time.time() > deadline:
                return False
            if row_idx == n_rows:
                return True
            
            # Generate candidate rows for this position
            # We need to be smart: don't generate all, just try promising ones
            
            # Strategy: try rows with smaller max entries first
            # Generate candidates incrementally
            
            # For efficiency, we'll generate a limited set of candidates
            # and try them in order of increasing max entry
            
            candidates = []
            # Generate up to 500 candidates
            cands = _generate_row_candidates(k, scope, used_diffs, deadline, limit=500)
            # Sort by max entry (last element)
            cands.sort(key=lambda x: x[0][-1])
            
            for row_tuple, diffs in cands:
                if time.time() > deadline:
                    return False
                
                # Check if adding this row keeps us within scope
                # (already ensured by generation)
                
                # Try this row
                rows.append(list(row_tuple))
                used_diffs.update(diffs)
                
                if backtrack(row_idx + 1, deadline):
                    return True
                
                rows.pop()
                used_diffs.difference_update(diffs)
            
            return False
        
        return backtrack(0, deadline) if backtrack(0, deadline) else None
    
    # Try scopes in increasing order from search_start to best_scope - 1
    # But we want to minimize scope, so try smaller scopes first
    # However, smaller scopes may have fewer candidates, making search harder
    # Trade-off: try a few small scopes, then jump to best_scope - 1
    
    scopes_to_try = list(range(search_start, best_scope))
    # Shuffle slightly to avoid getting stuck, but keep small scopes first
    # Actually, let's try in order: small scopes first
    
    for scope in scopes_to_try:
        if time.time() > deadline - 5:
            break
        
        # Check counting bound: can we fit all diffs in [1..scope]?
        if scope < n_rows * diffs_per_row:
            continue
        
        result = _bbs_inner(scope, deadline - 5)
        if result is not None:
            if _valid(result, n_rows, k):
                best_rows = result
                best_scope = scope
                break  # Found the minimum scope
    
    return best_rows, best_scope

def _local_search(rows, n_rows, k, rng, deadline):
    """Local search to improve scope."""
    if rows is None:
        return None, float('inf')
    
    cur = [r[:] for r in rows]
    cur_val = _max_entry(cur)
    used = set()
    for r in cur:
        used.update(_row_diffs_set(r))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        target = max(range(n_rows), key=lambda i: cur[i][-1])
        old_row = cur[target]
        old_diffs = _row_diffs_set(old_row)
        used -= old_diffs
        
        best_new = None
        for attempt in range(200):
            if time.time() > deadline:
                break
            new_row = [0]
            new_diffs = set()
            target_max = old_row[-1] - 1 - attempt // 20
            if target_max < k:
                break
            ok = True
            for _pos in range(k):
                found = False
                lo = new_row[-1] + 1
                hi = target_max - (k - len(new_row))
                if hi < lo:
                    ok = False
                    break
                for _trial in range(50):
                    v = lo + rng.randint(0, max(0, hi - lo))
                    nd = set()
                    valid = True
                    for a in new_row:
                        d = v - a
                        if d in new_diffs or d in used or d in nd:
                            valid = False
                            break
                        nd.add(d)
                    if valid:
                        new_row.append(v)
                        new_diffs.update(nd)
                        found = True
                        break
                if not found:
                    ok = False
                    break
            if ok and new_row[-1] < old_row[-1]:
                if best_new is None or new_row[-1] < best_new[-1]:
                    best_new = new_row
        
        if best_new is not None:
            cur[target] = best_new
            used.update(_row_diffs_set(best_new))
            new_val = _max_entry(cur)
            if new_val < cur_val:
                cur_val = new_val
                improved = True
        else:
            used |= old_diffs
    
    return cur, cur_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    rng = random.Random(42)
    
    # Phase 1: B&B search
    rows, scope = _bbs_search(n_rows, k, deadline - 10)
    
    # Phase 2: Local search refinement
    if rows is not None and time.time() < deadline - 5:
        improved_rows, improved_val = _local_search(rows, n_rows, k, rng, deadline - 5)
        if improved_rows is not None and improved_val < scope:
            rows = improved_rows
            scope = improved_val
    
    # Phase 3: If B&B didn't find anything, use greedy fallback
    if rows is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                nd = {x - a for a in row}
                if not (nd & used) and len(nd) == len(row):
                    row.append(x)
                    used |= nd
            rows.append(row)
        if _valid(rows, n_rows, k):
            scope = _max_entry(rows)
        else:
            # Last resort: return empty or invalid
            return []
    
    return rows

# EVOLVE-BLOCK-END
