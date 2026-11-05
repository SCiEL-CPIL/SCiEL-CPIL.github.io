"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _validate(rows, k):
    """Check that all positive differences within rows are distinct across all rows."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d <= 0:
                    return False
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k):
    """Greedy construction: each row takes the smallest next mark with unused differences."""
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


def _simulated_annealing(n_rows, k, deadline, seed=42):
    """
    Simulated Annealing over the set of rows.
    State: list of rows, each a sorted list of k+1 integers starting at 0.
    Moves: pick a random row, pick a random position, shift it by a random delta,
           then re-sort and re-validate.
    Objective: minimize scope (largest entry) subject to validity.
    We track the best valid solution found.
    """
    rng = random.Random(seed)
    best_rows = _greedy_construct(n_rows, k)
    best_scope = _scope(best_rows)
    
    # Compute the set of used differences for the current best
    best_used = set()
    for row in best_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                best_used.add(row[j] - row[i])
    
    # Start SA from a slightly perturbed version of greedy
    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope
    
    t_start = time.time()
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    step_size_initial = max(1, best_scope // 10)
    
    # Precompute: for a given row, the set of diffs it contributes
    def row_diffs(row):
        s = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                s.add(row[j] - row[i])
        return s
    
    # Build the global used set for current state
    current_used = set()
    for row in current_rows:
        current_used |= row_diffs(row)
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Choose a move
        r_idx = rng.randrange(n_rows)
        row = current_rows[r_idx]
        
        # Pick a position to perturb (not position 0)
        pos = rng.randrange(1, len(row))
        delta = rng.randint(-step_size_initial, step_size_initial)
        old_val = row[pos]
        new_val = old_val + delta
        
        if new_val <= 0:
            continue
        
        # Try the move
        row[pos] = new_val
        row.sort()
        
        # Check validity of the modified row
        row_d = row_diffs(row)
        if len(row_d) != len(row) * (len(row) - 1) // 2:
            # Duplicate differences within the row
            row[pos] = old_val
            row.sort()
            continue
        
        # Check against other rows' differences
        # Remove old row's diffs from current_used, check new
        # Actually, let's just check intersection with (current_used - old_row_diffs)
        other_used = current_used - row_diffs([old_val] if False else [row[:pos] + [old_val] + row[pos+1:]])
        # Simpler: compute other_used as all diffs not from this row
        # Let's recompute: temporarily set row back, get other_used
        row[pos] = old_val
        row.sort()
        old_row_d = row_diffs(row)
        other_used = current_used - old_row_d
        
        if row_d & other_used:
            # Conflict
            continue
        
        # Accept the move
        row[pos] = new_val
        row.sort()
        current_used = other_used | row_d
        new_scope = max(max(r) for r in current_rows)
        
        # Update best if improved
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [list(r) for r in current_rows]
            best_used = set(current_used)
            # Reduce step size as we improve
            step_size_initial = max(1, best_scope // 15)
        
        # Cooling
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Occasionally reset to best to avoid getting stuck
        if iteration % 500 == 0 and new_scope > best_scope:
            current_rows = [list(r) for r in best_rows]
            current_used = set(best_used)
            current_scope = best_scope
    
    return best_rows


def _local_search_refine(rows, k, deadline, seed=123):
    """
    Local search: try to reduce the scope by moving individual elements.
    For each row, for each position, try to decrease the value while maintaining validity.
    """
    rng = random.Random(seed)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    def row_diffs(row):
        s = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                s.add(row[j] - row[i])
        return s
    
    current_rows = [list(r) for r in best_rows]
    current_used = set()
    for row in current_rows:
        current_used |= row_diffs(row)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to reduce each element
        for r_idx in range(len(current_rows)):
            row = current_rows[r_idx]
            for pos in range(1, len(row)):
                old_val = row[pos]
                # Try decreasing
                for new_val in range(old_val - 1, 0, -1):
                    row[pos] = new_val
                    row.sort()
                    row_d = row_diffs(row)
                    if len(row_d) != len(row) * (len(row) - 1) // 2:
                        continue
                    old_row_d = row_diffs([row[:pos] + [old_val] + row[pos+1:]])
                    other_used = current_used - old_row_d
                    if row_d & other_used:
                        continue
                    # Valid move
                    current_used = other_used | row_d
                    new_scope = max(max(r) for r in current_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in current_rows]
                    improved = True
                    break
                else:
                    row[pos] = old_val
                    row.sort()
    
    return best_rows


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry (scope) such that all positive differences
    within rows are distinct across all rows.
    """
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill
    
    # Start with greedy
    rows = _greedy_construct(n_rows, k)
    scope = _scope(rows)
    
    # If greedy is already very good (small scope), we might not need SA
    # But let's try SA to improve
    
    # Run simulated annealing
    rows_sa = _simulated_annealing(n_rows, k, deadline, seed=42)
    scope_sa = _scope(rows_sa)
    
    if scope_sa < scope:
        rows = rows_sa
        scope = scope_sa
    
    # Run local search to refine
    rows_ls = _local_search_refine(rows, k, deadline, seed=123)
    scope_ls = _scope(rows_ls)
    
    if scope_ls < scope:
        rows = rows_ls
        scope = scope_ls
    
    # Validate final result
    if not _validate(rows, k):
        # Fallback to greedy which is guaranteed valid
        rows = _greedy_construct(n_rows, k)
    
    return rows

# EVOLVE-BLOCK-END
