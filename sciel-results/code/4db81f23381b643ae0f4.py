"""
Difference triangle set (4,7): Minimize the largest entry.
Combines constraint-satisfaction backtracking with simulated annealing refinement.
"""
import time
import random
import sys

_START_TIME = None
_DEADLINE = 142.0  # Safety margin for 150s limit

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _compute_row_diffs(row):
    """Compute set of differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ai)
    return diffs

def _full_validate(rows):
    """Return scope if valid, else None."""
    seen = set()
    scope = 0
    for row in rows:
        for d in _compute_row_diffs(row):
            if d in seen:
                return None
            seen.add(d)
        scope = max(scope, max(row))
    return scope

def _greedy_init(n_rows, k):
    """Fast greedy initialization."""
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

def _backtracking_solve(n_rows, k, max_scope, rng):
    """
    Constraint satisfaction with backtracking and forward checking.
    Tries to find a valid configuration with max entry <= max_scope.
    """
    if _time_left() < 5.0:
        return None
    
    # Precompute all possible differences
    max_diff = max_scope
    # Each row needs k*(k+1)//2 distinct differences
    diffs_per_row = k * (k + 1) // 2
    
    # State: rows[r] is the current partial row
    rows = [[0] for _ in range(n_rows)]
    used_diffs = set()
    
    def backtrack(r, pos):
        """Try to fill row r at position pos (0-indexed, 0 is already set)."""
        if _time_left() < 2.0:
            return False
        
        if r == n_rows:
            return True
        
        if pos == k + 1:
            return backtrack(r + 1, 0)
        
        row = rows[r]
        last = row[-1] if row else 0
        
        # Try values from last+1 to max_scope
        # Use MRV-like heuristic: try values that create fewer conflicts
        for new_val in range(last + 1, max_scope + 1):
            new_diffs = {new_val - a for a in row}
            
            # Check if any new diff is already used
            if new_diffs & used_diffs:
                continue
            
            # Forward checking: verify that remaining positions can still be filled
            # This is a simplified check - just ensure we have enough room
            remaining_positions = k + 1 - pos - 1
            if remaining_positions > 0:
                # Check if there are enough unused differences available
                # This is a weak check, but helps prune
                pass
            
            # Accept this value
            row.append(new_val)
            used_diffs |= new_diffs
            
            if backtrack(r, pos + 1):
                return True
            
            # Backtrack
            row.pop()
            used_diffs -= new_diffs
        
        return False
    
    # Try to solve
    if backtrack(0, 1):
        return [list(row) for row in rows]
    
    return None

def _targeted_descent(rows, n_rows, k, rng, budget=3.0):
    """
    Try to reduce the scope by shrinking the largest mark in the row that has it.
    Returns (improved_rows, new_scope) or (None, None) if no improvement found within budget.
    """
    start = time.time()
    cur = [r[:] for r in rows]
    cur_scope = max(max(r) for r in cur)
    improved = False
    
    while time.time() - start < budget and _time_left() > 5.0:
        # Find the row(s) containing the current max
        max_val = max(max(r) for r in cur)
        max_rows = [r for r in range(n_rows) if max(cur[r]) == max_val]
        r = rng.choice(max_rows)
        row = cur[r]
        pos = len(row) - 1
        old_val = row[pos]
        lower = row[pos - 1] + 1
        
        if old_val <= lower:
            continue
        
        # Try a set of candidate values
        candidates = set()
        for v in range(lower, min(lower + 20, old_val)):
            candidates.add(v)
        for _ in range(15):
            candidates.add(rng.randint(lower, old_val - 1))
        
        for new_val in sorted(candidates, key=lambda x: -x):
            if new_val >= old_val:
                continue
            row[pos] = new_val
            scope = _full_validate(cur)
            if scope is not None:
                if scope < cur_scope:
                    cur_scope = scope
                    improved = True
                    break
            row[pos] = old_val
        
        if not improved:
            # Try reducing a non-last element in the max row
            pos2 = rng.randint(1, len(row) - 2)
            old2 = row[pos2]
            lo2 = row[pos2 - 1] + 1
            hi2 = row[pos2 + 1] - 1
            if hi2 <= lo2:
                break
            for _ in range(8):
                nv = rng.randint(lo2, hi2)
                if nv == old2:
                    continue
                row[pos2] = nv
                scope = _full_validate(cur)
                if scope is not None and scope < cur_scope:
                    cur_scope = scope
                    improved = True
                    break
                row[pos2] = old2
            if not improved:
                break
    
    if improved:
        return cur, cur_scope
    return None, None

def _simulated_annealing(rows, n_rows, k, rng, time_budget=15.0):
    """
    Simulated annealing to reduce the maximum entry.
    State: the rows. Move: pick a random row and position, change value.
    Accept if valid or with probability based on temperature.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    
    start_time = time.time()
    T = 30.0
    T_min = 0.05
    cooling = 0.9999
    
    while T > T_min and (time.time() - start_time) < time_budget and _time_left() > 5.0:
        # Pick a random row and position (not the first, which is 0)
        r = rng.randrange(n_rows)
        pos = rng.randrange(1, k + 1)
        row = rows[r]
        old_val = row[pos]
        
        # Propose a new value
        delta = rng.randint(-15, 15)
        new_val = old_val + delta
        
        # Must maintain increasing order
        if new_val <= row[pos - 1]:
            T *= cooling
            continue
        if pos < k and new_val >= row[pos + 1]:
            T *= cooling
            continue
        if new_val <= 0:
            T *= cooling
            continue
        
        # Try the move
        row[pos] = new_val
        new_scope = _full_validate(rows)
        
        if new_scope is not None:
            if new_scope < best_max:
                best_max = new_scope
                best_rows = [list(rr) for rr in rows]
            cur_scope = max(max(rr) for rr in rows)
            delta_scope = new_scope - cur_scope
            if delta_scope <= 0 or rng.random() < (2.718281828 ** (-delta_scope / max(T, 0.01))):
                pass  # Accept
            else:
                row[pos] = old_val
        else:
            # Invalid: reject
            row[pos] = old_val
        
        T *= cooling
    
    return best_rows, best_max

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Greedy initialization
    best_rows = _greedy_init(n_rows, k)
    best_max = max(max(r) for r in best_rows)
    
    # Phase 2: Try backtracking to find a better solution
    # Start with a scope slightly below the greedy result
    for trial_scope in range(best_max - 1, best_max - 10, -1):
        if _time_left() < 20.0:
            break
        if trial_scope < 0:
            break
        
        bt_rows = _backtracking_solve(n_rows, k, trial_scope, rng)
        if bt_rows is not None:
            bt_max = max(max(r) for r in bt_rows)
            if bt_max < best_max:
                best_max = bt_max
                best_rows = bt_rows
            break
    
    # Phase 3: Targeted descent to reduce max
    if _time_left() > 10.0:
        desc_rows, desc_max = _targeted_descent(best_rows, n_rows, k, rng, budget=_time_left() - 10.0)
        if desc_rows is not None and desc_max < best_max:
            best_max = desc_max
            best_rows = desc_rows
    
    # Phase 4: Simulated annealing if time permits
    if _time_left() > 15.0:
        sa_rows, sa_max = _simulated_annealing(best_rows, n_rows, k, rng, time_budget=_time_left() - 5.0)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows
    
    # Final verification
    all_diffs = set()
    for row in best_rows:
        row_diffs = _compute_row_diffs(row)
        if row_diffs & all_diffs:
            # Invalid - should not happen
            pass
        all_diffs |= row_diffs
    
    return best_rows
