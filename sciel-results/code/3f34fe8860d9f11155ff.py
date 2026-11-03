"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _compute_scope(rows):
    """Return the largest entry (scope) across all rows."""
    return max(max(r) for r in rows)

def _is_valid(rows):
    """Check if all positive differences within rows are distinct across all rows."""
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _greedy_construct(n_rows, k, rng=None):
    """Greedy construction: each row takes smallest next mark with no diff conflicts."""
    if rng is None:
        rng = random.Random(42)
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

def _sa_improve(rows, n_rows, k, time_budget, rng):
    """
    Simulated annealing to reduce the scope.
    Moves: pick a random row, pick a random non-zero element, try to shift it
    to a smaller or different position, maintaining strict increase and no diff conflicts.
    Objective: minimize the maximum entry (scope).
    """
    def get_scope():
        return max(max(r) for r in rows)
    
    def get_all_diffs(row):
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Precompute all diffs for each row
    all_row_diffs = [get_all_diffs(r) for r in rows]
    used_diffs = set()
    for rd in all_row_diffs:
        used_diffs |= rd
    
    current_scope = get_scope()
    best_rows = [list(r) for r in rows]
    best_scope = current_scope
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Temperature schedule
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    
    while time.time() < deadline:
        # Pick a random row and a random element to move
        ri = rng.randint(0, n_rows - 1)
        # Pick an element index (1..k, since index 0 is fixed at 0)
        ei = rng.randint(1, k)
        
        old_val = rows[ri][ei]
        
        # Try to move to a new value
        # Bias towards smaller values to reduce scope
        if rng.random() < 0.6:
            # Try smaller value
            new_val = old_val - rng.randint(1, max(1, old_val // 3))
            new_val = max(1, new_val)
        else:
            # Try random nearby value
            new_val = old_val + rng.randint(-5, 5)
            new_val = max(1, new_val)
        
        if new_val == old_val:
            continue
        
        # Check if new value maintains strict increase
        row = rows[ri]
        if ei > 0 and new_val <= row[ei - 1]:
            continue
        if ei < k and new_val >= row[ei + 1]:
            continue
        
        # Check diff conflicts
        old_diffs = set()
        for j in range(k + 1):
            if j != ei:
                d = abs(new_val - row[j]) if new_val != row[j] else 0
                # Actually compute diffs involving ei
        # Better: compute new diffs for row ri
        new_row = row[:]
        new_row[ei] = new_val
        
        # Check strict increase
        for j in range(k):
            if new_row[j] >= new_row[j + 1]:
                continue
        
        # Compute new diffs for this row
        new_diffs = set()
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                new_diffs.add(new_row[j] - new_row[i])
        
        # Diffs that are in new_row but not in old row
        added_diffs = new_diffs - all_row_diffs[ri]
        # Check if any added diff conflicts with other rows
        conflict = False
        for d in added_diffs:
            if d in used_diffs and d not in all_row_diffs[ri]:
                # This diff is used by another row
                conflict = True
                break
            if d in all_row_diffs[ri]:
                # Still used by this row (from other pairs), that's fine
                pass
        
        # More careful: a diff is valid if it's not used by any OTHER row
        # and within this row, it's fine to have it used by multiple pairs? No!
        # The constraint is ALL differences across ALL rows must be distinct.
        # So within a single row, each pair gives a diff, and ALL these diffs
        # across all rows must be unique.
        
        # So: new_diffs must not intersect with (used_diffs - all_row_diffs[ri])
        # because used_diffs - all_row_diffs[ri] are diffs used by other rows.
        other_diffs = used_diffs - all_row_diffs[ri]
        if new_diffs & other_diffs:
            continue  # Conflict with other rows
        
        # Accept the move
        # Update state
        removed_diffs = all_row_diffs[ri] - new_diffs
        added_diffs_final = new_diffs - all_row_diffs[ri]
        
        for d in removed_diffs:
            used_diffs.discard(d)
        for d in added_diffs_final:
            used_diffs.add(d)
        
        rows[ri] = new_row
        all_row_diffs[ri] = new_diffs
        
        new_scope = get_scope()
        delta_scope = new_scope - current_scope
        
        if delta_scope <= 0 or rng.random() < pow(2.718281828, -delta_scope / max(T, 1e-9)):
            current_scope = new_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
        else:
            # Reject
            for d in added_diffs_final:
                used_diffs.discard(d)
            for d in removed_diffs:
                used_diffs.add(d)
            rows[ri][ei] = old_val
            all_row_diffs[ri] = get_all_diffs(rows[ri])
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_rows, best_scope

def _polish(rows, n_rows, k, time_budget, rng):
    """
    Local search polish: try to reduce each element to the smallest valid value.
    Iterate over all rows and elements, trying to shrink them.
    """
    def get_all_diffs(row):
        n = len(row)
        diffs = set()
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs
    
    all_row_diffs = [get_all_diffs(r) for r in rows]
    used_diffs = set()
    for rd in all_row_diffs:
        used_diffs |= rd
    
    start_time = time.time()
    deadline = start_time + time_budget
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for ri in range(n_rows):
            for ei in range(1, k + 1):  # Skip index 0 (fixed at 0)
                row = rows[ri]
                old_val = row[ei]
                
                # Try to reduce to smallest valid value
                for new_val in range(row[ei - 1] + 1, old_val):
                    new_row = row[:]
                    new_row[ei] = new_val
                    new_diffs = get_all_diffs(new_row)
                    other_diffs = used_diffs - all_row_diffs[ri]
                    if new_diffs & other_diffs:
                        continue
                    # Accept
                    removed = all_row_diffs[ri] - new_diffs
                    added = new_diffs - all_row_diffs[ri]
                    for d in removed:
                        used_diffs.discard(d)
                    for d in added:
                        used_diffs.add(d)
                    rows[ri] = new_row
                    all_row_diffs[ri] = new_diffs
                    improved = True
                    break  # Move to next element
    
    return rows

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, such that all positive differences
    within rows are distinct across all rows.
    """
    start_time = time.time()
    total_deadline = 130.0  # Safety margin before 150s kill
    
    # Phase 1: Greedy construction
    rng = random.Random(12345)
    rows = _greedy_construct(n_rows, k, rng)
    scope = _compute_scope(rows)
    
    elapsed = time.time() - start_time
    if elapsed > total_deadline - 5:
        return rows
    
    # Phase 2: Simulated annealing to improve
    sa_budget = min(60.0, total_deadline - elapsed - 10)
    if sa_budget > 5:
        rows, scope = _sa_improve(rows, n_rows, k, sa_budget, rng)
    
    elapsed = time.time() - start_time
    if elapsed > total_deadline - 5:
        return rows
    
    # Phase 3: Polish
    polish_budget = min(30.0, total_deadline - elapsed - 2)
    if polish_budget > 1:
        rows = _polish(rows, n_rows, k, polish_budget, rng)
    
    # Final validation
    if not _is_valid(rows):
        # Fallback to greedy
        rows = _greedy_construct(n_rows, k, random.Random(999))
    
    return rows

# EVOLVE-BLOCK-END
