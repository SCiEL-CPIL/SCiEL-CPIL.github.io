"""Improved program for difference triangle set (4,7)."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of positive differences for a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _greedy_row(used, k, max_val=200):
    """Build a single row greedily: pick smallest next x such that no new diff conflicts."""
    row = [0]
    while len(row) < k + 1:
        found = False
        for x in range(1, max_val + 1):
            new_diffs = set()
            for a in row:
                d = x - a
                if d in used:
                    break
                new_diffs.add(d)
            else:
                row.append(x)
                used |= new_diffs
                found = True
                break
        if not found:
            return None
    return row

def _validate(rows):
    """Check that all differences across all rows are distinct."""
    all_diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return True

def _score(rows):
    """Return the largest entry (scope)."""
    return max(max(r) for r in rows)

def _construct_greedy(n_rows, k):
    """Original greedy approach."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        if row is None:
            break
        rows.append(row)
    if len(rows) < n_rows:
        return None
    return rows

def _construct_sa(n_rows, k, time_limit=100):
    """Simulated annealing to minimize the scope."""
    deadline = time.time() + time_limit
    rng = random.Random(42)
    
    # Start from greedy
    rows = _construct_greedy(n_rows, k)
    if rows is None:
        return None
    
    current_scope = _score(rows)
    best_rows = [r[:] for r in rows]
    best_scope = current_scope
    
    used = set()
    for row in rows:
        used |= _row_diffs(row)
    
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    
    while time.time() < deadline and T > T_min:
        # Pick a random row and a random position to perturb
        ri = rng.randint(0, n_rows - 1)
        pi = rng.randint(1, k)  # positions 1..k (not 0)
        
        old_val = rows[ri][pi]
        
        # Try to move the value by a small amount
        delta = rng.choice([-3, -2, -1, 1, 2, 3])
        new_val = old_val + delta
        
        if new_val <= 0:
            continue
        if pi > 0 and new_val >= rows[ri][pi - 1]:
            continue
        if pi < k and new_val > rows[ri][pi + 1]:
            continue
        if new_val == old_val:
            continue
        
        # Compute new diffs for this row
        new_row = rows[ri][:]
        new_row[pi] = new_val
        new_diffs = _row_diffs(new_row)
        old_diffs = _row_diffs(rows[ri])
        
        # Check conflicts with other rows' diffs
        other_diffs = used - old_diffs
        if new_diffs & other_diffs:
            T *= cooling
            continue
        
        # Accept
        rows[ri] = new_row
        used = (used - old_diffs) | new_diffs
        current_scope = _score(rows)
        
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [r[:] for r in rows]
        
        T *= cooling
    
    return best_rows

def _construct_backtracking(n_rows, k, time_limit=100):
    """Backtracking search with constraint propagation."""
    deadline = time.time() + time_limit
    rng = random.Random(123)
    
    best_rows = None
    best_scope = float('inf')
    
    # First get a greedy solution as upper bound
    greedy = _construct_greedy(n_rows, k)
    if greedy is not None:
        best_scope = _score(greedy)
        best_rows = [r[:] for r in greedy]
    
    used = set()
    rows = []
    
    def backtrack(row_idx, row, current_max):
        nonlocal best_scope, best_rows
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            if current_max < best_scope:
                best_scope = current_max
                best_rows = [r[:] for r in rows]
            return True
        
        if len(row) == 0:
            row = [0]
        
        # Try to complete this row
        def complete_row(row, used, depth=0):
            nonlocal deadline
            if time.time() > deadline:
                return None
            if len(row) == k + 1:
                return row
            # Greedy with some randomization
            candidates = []
            for x in range(row[-1] + 1, best_scope + 1):
                new_diffs = set()
                ok = True
                for a in row:
                    d = x - a
                    if d in used:
                        ok = False
                        break
                    new_diffs.add(d)
                if ok:
                    candidates.append((x, new_diffs))
            
            if not candidates:
                return None
            
            # Try candidates in order (greedy) or shuffled
            if depth < 3:
                for x, new_diffs in candidates[:5]:
                    row.append(x)
                    used |= new_diffs
                    result = complete_row(row, used, depth + 1)
                    if result is not None:
                        return result
                    used -= new_diffs
                    row.pop()
            else:
                # Just take the first valid one
                x, new_diffs = candidates[0]
                row.append(x)
                used |= new_diffs
                return complete_row(row, used, depth + 1)
            return None
        
        completed = complete_row(row, used)
        if completed is not None:
            rows.append(completed)
            if backtrack(row_idx + 1, [], max(best_scope, max(completed))):
                return True
            rows.pop()
        
        return False
    
    backtrack(0, [], best_scope)
    
    return best_rows

def _construct_hybrid(n_rows, k, time_limit=110):
    """Combine greedy, SA, and backtracking."""
    deadline = time.time() + time_limit
    
    best_rows = None
    best_scope = float('inf')
    
    # 1. Greedy
    greedy = _construct_greedy(n_rows, k)
    if greedy is not None:
        best_scope = _score(greedy)
        best_rows = [r[:] for r in greedy]
    
    # 2. SA if time permits
    if time.time() < deadline - 20:
        sa_rows = _construct_sa(n_rows, k, time_limit=min(40, deadline - time.time() - 20))
        if sa_rows is not None:
            sa_scope = _score(sa_rows)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = [r[:] for r in sa_rows]
    
    # 3. Backtracking if time permits
    if time.time() < deadline - 10:
        bt_rows = _construct_backtracking(n_rows, k, time_limit=min(30, deadline - time.time() - 10))
        if bt_rows is not None:
            bt_scope = _score(bt_rows)
            if bt_scope < best_scope:
                best_scope = bt_scope
                best_rows = [r[:] for r in bt_rows]
    
    return best_rows

def construct(n_rows, k):
    """Main entry point: construct n_rows rows of k+1 increasing integers starting at 0."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # For the specific task (4,7), use hybrid approach
    if n_rows == 4 and k == 7:
        result = _construct_hybrid(n_rows, k, time_limit=110)
    else:
        # General case: try greedy first, then SA
        result = _construct_greedy(n_rows, k)
        if result is None or time.time() < deadline - 20:
            sa_result = _construct_sa(n_rows, k, time_limit=min(50, deadline - time.time() - 5))
            if sa_result is not None:
                if result is None or _score(sa_result) < _score(result):
                    result = sa_result
    
    if result is None:
        # Fallback: return a trivial valid (but suboptimal) solution
        # This shouldn't happen for reasonable inputs
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                x += 1
            rows.append(row)
        return rows
    
    # Final validation
    if not _validate(result):
        # Fallback to greedy
        fallback = _construct_greedy(n_rows, k)
        if fallback is not None and _validate(fallback):
            return fallback
    
    return result

# EVOLVE-BLOCK-END
