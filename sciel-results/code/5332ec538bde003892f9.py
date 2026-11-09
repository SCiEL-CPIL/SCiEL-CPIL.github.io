"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _validate(rows, k):
    """Check if rows form a valid difference triangle set."""
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
    return max(max(row) for row in rows)

def _greedy_row(used, k, start=1):
    """Build a single row greedily using the smallest available differences."""
    row = [0]
    x = start
    while len(row) < k + 1:
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
        x += 1
    return row

def _greedy_construct(n_rows, k):
    """Build all rows greedily, one at a time."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows

def _local_search(rows, k, time_limit=10.0, seed=42):
    """
    Local search to reduce scope.
    Try to replace individual row elements with smaller values that maintain validity.
    """
    rng = random.Random(seed)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    start_time = time.time()
    
    # Precompute: for a given row and position, what values are possible?
    # We'll use a simpler approach: try to shrink the last element first, then others.
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try each row, each position from the end
        for r_idx in range(len(best_rows)):
            row = best_rows[r_idx]
            for pos in range(len(row) - 1, 0, -1):
                old_val = row[pos]
                # Try smaller values
                for new_val in range(row[pos - 1] + 1, old_val):
                    row[pos] = new_val
                    # Check validity of this row with other rows
                    if _validate(best_rows, k):
                        new_scope = _scope(best_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            improved = True
                            break
                    row[pos] = old_val
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Try random perturbations
            r_idx = rng.randint(0, len(best_rows) - 1)
            pos = rng.randint(1, len(best_rows[r_idx]) - 1)
            old_val = best_rows[r_idx][pos]
            # Try a random smaller value
            if old_val > best_rows[r_idx][pos - 1] + 1:
                new_val = rng.randint(best_rows[r_idx][pos - 1] + 1, old_val - 1)
                best_rows[r_idx][pos] = new_val
                if _validate(best_rows, k):
                    new_scope = _scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        improved = True
                else:
                    best_rows[r_idx][pos] = old_val
            if not improved:
                # Try increasing a value to allow others to decrease (not directly helpful for scope)
                # Or try swapping positions
                break
    return best_rows

def _backtrack_row(used, k, max_val, row_prefix, pos):
    """
    Backtrack to find a row of length k+1 starting with row_prefix,
    using differences not in 'used', with all values <= max_val.
    Returns the row or None.
    """
    if pos == k + 1:
        return list(row_prefix)
    
    # Try values from row_prefix[-1] + 1 to max_val
    for x in range(row_prefix[-1] + 1, max_val + 1):
        new_diffs = {x - a for a in row_prefix}
        if not new_diffs & used:
            row_prefix.append(x)
            used |= new_diffs
            result = _backtrack_row(used, k, max_val, row_prefix, pos + 1)
            if result is not None:
                return result
            used -= new_diffs
            row_prefix.pop()
    return None

def _construct_with_max_scope(n_rows, k, max_scope, time_limit=30.0):
    """
    Try to construct n_rows rows with all values <= max_scope.
    Uses backtracking for each row.
    """
    used = set()
    rows = []
    start_time = time.time()
    
    for r in range(n_rows):
        if time.time() - start_time > time_limit:
            return None
        # Try to build row r with backtracking
        row = _backtrack_row(used, k, max_scope, [0], 1)
        if row is None:
            return None
        rows.append(row)
        used |= {row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))}
    
    return rows

def _binary_search_max_scope(n_rows, k, low, high, time_limit=60.0):
    """Binary search for the minimum max_scope that allows a valid construction."""
    best_result = None
    start_time = time.time()
    
    while low <= high and time.time() - start_time < time_limit:
        mid = (low + high) // 2
        result = _construct_with_max_scope(n_rows, k, mid, time_limit=max(1.0, time_limit - (time.time() - start_time)))
        if result is not None:
            best_result = result
            high = mid - 1
        else:
            low = mid + 1
    
    return best_result

def _simulated_annealing(rows, k, time_limit=20.0, seed=42):
    """
    Simulated annealing to reduce scope.
    Moves: change one element in one row to a nearby value.
    """
    rng = random.Random(seed)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    current_rows = [list(r) for r in rows]
    current_scope = best_scope
    start_time = time.time()
    
    T = 10.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Make a random move
        r_idx = rng.randint(0, len(current_rows) - 1)
        pos = rng.randint(1, len(current_rows[r_idx]) - 1)
        old_val = current_rows[r_idx][pos]
        # Try a random perturbation
        delta = rng.randint(-5, 5)
        new_val = old_val + delta
        # Ensure increasing
        if new_val <= current_rows[r_idx][pos - 1]:
            new_val = current_rows[r_idx][pos - 1] + 1
        if new_val > 200:  # Cap to avoid huge values
            new_val = old_val
        
        current_rows[r_idx][pos] = new_val
        if _validate(current_rows, k):
            new_scope = _scope(current_rows)
            delta_scope = new_scope - current_scope
            if delta_scope < 0 or rng.random() < pow(2.718281828, -delta_scope / T):
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                current_rows[r_idx][pos] = old_val
        else:
            current_rows[r_idx][pos] = old_val
        
        T *= cooling
        if T < 0.1:
            break
    
    return best_rows

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows.
    Minimize the largest entry.
    """
    deadline = time.time() + 145.0  # Safety margin
    
    # Phase 1: Greedy construction
    rows = _greedy_construct(n_rows, k)
    best_scope = _scope(rows)
    
    # Phase 2: Local search
    if time.time() < deadline - 30:
        rows_ls = _local_search(rows, k, time_limit=min(15.0, deadline - time.time() - 30), seed=42)
        if _scope(rows_ls) < best_scope:
            rows = rows_ls
            best_scope = _scope(rows)
    
    # Phase 3: Binary search with backtracking for tighter scope
    if time.time() < deadline - 20:
        # Start from a reasonable lower bound
        low = k * (k + 1) // 2  # Simple lower bound
        high = best_scope
        result = _binary_search_max_scope(n_rows, k, low, high, time_limit=min(40.0, deadline - time.time() - 20))
        if result is not None:
            new_scope = _scope(result)
            if new_scope < best_scope:
                rows = result
                best_scope = new_scope
    
    # Phase 4: Simulated annealing for further improvement
    if time.time() < deadline - 10:
        rows_sa = _simulated_annealing(rows, k, time_limit=min(20.0, deadline - time.time() - 10), seed=42)
        if _scope(rows_sa) < best_scope:
            rows = rows_sa
            best_scope = _scope(rows)
    
    # Final validation
    if not _validate(rows, k):
        # Fallback to greedy
        rows = _greedy_construct(n_rows, k)
    
    return rows

# EVOLVE-BLOCK-END
