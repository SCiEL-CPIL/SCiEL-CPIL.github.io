"""Difference triangle set (7,5) - CSP backtracking with difference-cost heuristic."""
import time
import random

_DEADLINE = 120.0  # safety margin under the 150s wall-clock limit


def _row_diffs(row):
    """All positive differences a_j - a_i for i < j in a single row."""
    d = []
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.append(row[j] - ai)
    return d


def _scope(rows):
    return max(max(r) for r in rows)


def _is_valid(rows):
    """Check if all differences across all rows are distinct."""
    used = set()
    for r in rows:
        for d in _row_diffs(r):
            if d in used:
                return False
            used.add(d)
    return True


def _greedy_baseline(n_rows, k):
    """Fast greedy construction to get a valid starting point."""
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


def _csp_backtrack(n_rows, k, deadline, initial_rows=None, scope_limit=None):
    """
    CSP backtracking: build rows one element at a time.
    State: (rows, used_diffs, current_row, current_pos)
    Heuristic: prefer smaller values that don't block many future differences.
    """
    if initial_rows is None:
        initial_rows = []
    if scope_limit is None:
        scope_limit = 200  # initial upper bound
    
    # We'll build rows incrementally. For each row, we place elements left to right.
    # The key insight: small differences are more "scarce" because they conflict
    # with more potential placements. So we track which differences are used and
    # prefer placements that use larger differences (less scarce).
    
    rows = [list(r) for r in initial_rows]
    used = set()
    for r in rows:
        for d in _row_diffs(r):
            used.add(d)
    
    best_scope = scope_limit
    best_rows = [list(r) for r in rows] if rows else None
    
    # If we already have some rows, continue from there
    next_row_idx = len(rows)
    
    def _cost_of_diffs(new_diffs, used_set):
        """Cost = sum of 1/d for each new difference. Smaller d = higher cost (more scarce)."""
        cost = 0.0
        for d in new_diffs:
            cost += 1.0 / d
        return cost
    
    def _dfs(current_row, current_x, row_idx, rows_so_far, used_so_far, deadline):
        nonlocal best_scope, best_rows
        
        if time.time() > deadline:
            return False
        
        # Check if we've completed all rows
        if row_idx == n_rows:
            scope = max(max(r) for r in rows_so_far)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows_so_far]
                return True
            return False
        
        # If this is a new row, start it
        if current_row is None:
            current_row = [0]
            current_x = 0
        
        # If current row is complete, move to next row
        if len(current_row) == k + 1:
            return _dfs(None, 0, row_idx + 1, rows_so_far + [current_row], used_so_far, deadline)
        
        # Try to place the next element in current_row
        # Candidate range: from current_x + 1 to some limit
        # Limit: we want to keep scope small, so don't go too far
        # But we need enough room for remaining elements
        remaining_elements = (k + 1) - len(current_row)
        
        # Estimate minimum additional span needed
        # Each remaining element needs at least 1 more, so min additional = remaining_elements
        # But we also need to avoid conflicts
        
        # Search window: try values from current_x + 1 upward
        # Cap the search to keep scope reasonable
        max_val = current_x + 50  # look-ahead window
        
        # But also respect scope_limit
        if best_rows is not None:
            # Current max in all completed rows
            if rows_so_far:
                existing_max = max(max(r) for r in rows_so_far)
            else:
                existing_max = 0
            # Don't exceed best_scope
            max_val = min(max_val, best_scope - 1)
        
        found = False
        for v in range(current_x + 1, max_val + 1):
            if time.time() > deadline:
                return False
            
            new_diffs = {v - a for a in current_row}
            if new_diffs & used_so_far:
                continue
            
            # Heuristic: prefer placements with lower cost (using larger differences)
            # But also prefer smaller v to keep scope tight
            # Combined score: v * 0.1 + cost
            cost = _cost_of_diffs(new_diffs, used_so_far)
            
            # Try this placement
            new_row = current_row + [v]
            new_used = used_so_far | new_diffs
            
            # Recursive call
            if _dfs(new_row, v, row_idx, rows_so_far, new_used, deadline):
                found = True
                # Don't return immediately; we want to find the best (smallest scope)
                # But for efficiency, we can return if we've hit a good bound
                if best_scope <= 105:  # counting bound
                    return True
        
        return found
    
    # Start the DFS
    # If we have initial rows, continue from there
    if initial_rows:
        _dfs(None, 0, next_row_idx, rows, used, deadline)
    else:
        _dfs(None, 0, 0, [], set(), deadline)
    
    if best_rows is not None:
        return best_rows, best_scope
    return None, None


def _local_search_shrink(rows, deadline):
    """
    Local search: try to reduce the scope by moving elements to smaller values.
    Works on a complete valid solution.
    """
    used = set()
    for r in rows:
        for d in _row_diffs(r):
            used.add(d)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(rows)):
            row = rows[i]
            # Try to move each element (except first) to a smaller value
            for pos in range(len(row) - 1, 0, -1):
                old_val = row[pos]
                
                # Compute all differences involving this element
                old_diffs = set()
                for j in range(pos):
                    old_diffs.add(old_val - row[j])
                for j in range(pos + 1, len(row)):
                    old_diffs.add(row[j] - old_val)
                
                # Remove from used
                used -= old_diffs
                
                # Try smaller values
                lower_bound = row[pos - 1] + 1
                found = False
                for v in range(old_val - 1, lower_bound - 1, -1):
                    cand_diffs = set()
                    for j in range(pos):
                        cand_diffs.add(v - row[j])
                    for j in range(pos + 1, len(row)):
                        cand_diffs.add(row[j] - v)
                    
                    if not (cand_diffs & used):
                        row[pos] = v
                        used |= cand_diffs
                        improved = True
                        found = True
                        break
                
                if not found:
                    used |= old_diffs
                
                if time.time() > deadline:
                    return rows
    
    return rows


def _simulated_annealing(rows, deadline):
    """
    Simulated annealing: random moves that may temporarily increase scope,
    accepting with probability exp(-delta/T).
    """
    used = set()
    for r in rows:
        for d in _row_diffs(r):
            used.add(d)
    
    current_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    best_scope = current_scope
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    rng = random.Random(42)
    
    while time.time() < deadline and T > T_min:
        # Pick a random row and element to move
        i = rng.randint(0, len(rows) - 1)
        row = rows[i]
        pos = rng.randint(1, len(row) - 1)
        
        old_val = row[pos]
        
        # Compute old diffs
        old_diffs = set()
        for j in range(pos):
            old_diffs.add(old_val - row[j])
        for j in range(pos + 1, len(row)):
            old_diffs.add(row[j] - old_val)
        
        used -= old_diffs
        
        # Try a random new value
        lower_bound = row[pos - 1] + 1
        upper_bound = old_val + 10  # can also increase slightly
        
        if upper_bound <= lower_bound:
            used |= old_diffs
            T *= cooling
            continue
        
        new_val = rng.randint(lower_bound, upper_bound)
        
        if new_val == old_val:
            used |= old_diffs
            T *= cooling
            continue
        
        cand_diffs = set()
        for j in range(pos):
            cand_diffs.add(new_val - row[j])
        for j in range(pos + 1, len(row)):
            cand_diffs.add(row[j] - new_val)
        
        if cand_diffs & used:
            used |= old_diffs
            T *= cooling
            continue
        
        # Valid move
        row[pos] = new_val
        used |= cand_diffs
        
        new_scope = _scope(rows)
        delta = new_scope - current_scope
        
        if delta <= 0 or rng.random() < (2.718281828 ** (-delta / T)):
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in rows]
        else:
            # Reject: revert
            row[pos] = old_val
            used -= cand_diffs
            used |= old_diffs
        
        T *= cooling
    
    return best_rows


def construct(n_rows, k):
    start = time.time()
    deadline = start + _DEADLINE - 5.0  # safety margin
    
    # Phase 1: Greedy baseline
    rows = _greedy_baseline(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    # Phase 2: Local search to shrink
    rows = _local_search_shrink(rows, deadline - 30.0)
    scope = _scope(rows)
    if scope < best_scope:
        best_rows = [list(r) for r in rows]
        best_scope = scope
    
    # Phase 3: CSP backtracking with the greedy solution as initial bound
    remaining = deadline - time.time()
    if remaining > 10.0:
        # Use the current best as the scope limit for backtracking
        bt_rows, bt_scope = _csp_backtrack(
            n_rows, k, deadline - 5.0,
            initial_rows=best_rows,
            scope_limit=best_scope
        )
        if bt_rows is not None and bt_scope < best_scope:
            best_rows = [list(r) for r in bt_rows]
            best_scope = bt_scope
    
    # Phase 4: More local search on the best solution
    remaining = deadline - time.time()
    if remaining > 5.0:
        rows = _local_search_shrink(best_rows, deadline - 3.0)
        scope = _scope(rows)
        if scope < best_scope:
            best_rows = [list(r) for r in rows]
            best_scope = scope
    
    # Phase 5: Simulated annealing for final polish
    remaining = deadline - time.time()
    if remaining > 5.0:
        rows = _simulated_annealing(best_rows, deadline - 3.0)
        scope = _scope(rows)
        if scope < best_scope:
            best_rows = [list(r) for r in rows]
            best_scope = scope
    
    # Final validation
    if not _is_valid(best_rows):
        # Fallback to greedy
        best_rows = _greedy_baseline(n_rows, k)
    
    return best_rows
