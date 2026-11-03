import time
import random
import sys
import math
import numpy as np

_DEADLINE = 148.0  # safety margin before 150s kill

def _scope(rows):
    """Return the largest entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)

def _valid(rows, k):
    """Check that all positive differences are distinct across all rows."""
    used = set()
    for row in rows:
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used or d in diffs:
                    return False
                diffs.add(d)
        used |= diffs
    return True

def _fast_greedy(n_rows, k):
    """Fast greedy: each row takes smallest next mark with unused diffs."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _build_diff_counts(rows):
    """
    Build a dictionary mapping each difference to the set of rows that use it.
    Also return a list of all differences used.
    """
    diff_to_rows = {}
    all_diffs = set()
    for idx, row in enumerate(rows):
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d not in diff_to_rows:
                    diff_to_rows[d] = set()
                diff_to_rows[d].add(idx)
                all_diffs.add(d)
    return diff_to_rows, all_diffs

def _remove_row_diffs(diff_to_rows, all_diffs, row_idx, row):
    """
    Remove all differences of a specific row from the global structures.
    Returns the set of differences removed.
    """
    removed = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            if d in diff_to_rows:
                diff_to_rows[d].discard(row_idx)
                if not diff_to_rows[d]:
                    del diff_to_rows[d]
                    all_diffs.discard(d)
                removed.add(d)
    return removed

def _add_row_diffs(diff_to_rows, all_diffs, row_idx, row):
    """
    Add all differences of a specific row to the global structures.
    Returns the set of differences added.
    """
    added = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d = row[j] - row[i]
            if d not in diff_to_rows:
                diff_to_rows[d] = set()
            diff_to_rows[d].add(row_idx)
            all_diffs.add(d)
            added.add(d)
    return added

def _row_rebuild(n_rows, k, row_idx, current_rows, diff_to_rows, all_diffs, rng, max_scope_hint=None):
    """
    Rebuild a single row (row_idx) from scratch, respecting global diff constraints.
    Returns the new row if successful, None otherwise.
    """
    # Remove current row's diffs
    old_row = current_rows[row_idx]
    removed_diffs = _remove_row_diffs(diff_to_rows, all_diffs, row_idx, old_row)
    
    # Rebuild row
    new_row = [0]
    x = 0
    success = False
    
    # Use a randomized greedy approach with backtracking for small k
    # For k=5, this is small enough
    def _build_row_recursive(current_row, max_val):
        if len(current_row) == k + 1:
            return current_row
        
        # Determine next value
        # We want to find a value v > current_row[-1] such that
        # all diffs v - a for a in current_row are not in all_diffs
        # and not in current_row's own diffs (which are unique by construction)
        
        # Heuristic: try values starting from current_row[-1] + 1
        # But we can be smarter: try a range
        
        # To keep it fast, we'll try a limited number of candidates
        # and use randomization if stuck
        
        start = current_row[-1] + 1
        # Limit search space to avoid explosion
        # We can estimate a reasonable upper bound
        
        # Try a few values in order, with some randomization
        # For efficiency, we'll try up to 20 candidates per step
        
        candidates = []
        # First, try sequential values
        for v in range(start, min(start + 50, max_val + 1)):
            candidates.append(v)
        
        # Add some random candidates
        if max_val > start + 50:
            for _ in range(10):
                v = rng.randint(start + 50, max_val)
                if v not in candidates:
                    candidates.append(v)
        
        rng.shuffle(candidates)
        
        for v in candidates:
            # Check if v creates any conflict
            conflict = False
            for a in current_row:
                d = v - a
                if d in all_diffs:
                    conflict = True
                    break
            
            if conflict:
                continue
            
            # No conflict, try this value
            new_row = current_row + [v]
            result = _build_row_recursive(new_row, max_val)
            if result is not None:
                return result
        
        return None
    
    # Set a reasonable max_val
    # Use the current max scope as a hint, but allow some flexibility
    if max_scope_hint is None:
        max_scope_hint = max(max(r) for r in current_rows) + 20
    else:
        max_scope_hint = min(max_scope_hint + 10, max_scope_hint + 20)
    
    # Try to build the row
    result = _build_row_recursive([0], max_scope_hint)
    
    if result is not None:
        # Add the new row's diffs
        _add_row_diffs(diff_to_rows, all_diffs, row_idx, result)
        return result
    else:
        # Restore old row's diffs
        _add_row_diffs(diff_to_rows, all_diffs, row_idx, old_row)
        return None

def _sa_optimize(n_rows, k, initial_rows, time_limit, rng):
    """
    Simulated annealing to minimize the scope (largest entry).
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Initialize
    current_rows = [r[:] for r in initial_rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    # Build diff structures
    diff_to_rows, all_diffs = _build_diff_counts(current_rows)
    
    # SA parameters
    T = 10.0  # Initial temperature
    cooling_rate = 0.9999
    min_T = 0.1
    
    # Precompute row indices
    row_indices = list(range(n_rows))
    
    while time.time() < deadline and T > min_T:
        # Pick a random row to rebuild
        row_idx = rng.choice(row_indices)
        
        # Try to rebuild the row
        # Use current scope as hint, but allow some expansion
        max_scope_hint = current_scope
        new_row = _row_rebuild(n_rows, k, row_idx, current_rows, diff_to_rows, all_diffs, rng, max_scope_hint)
        
        if new_row is not None:
            new_scope = _scope(current_rows)
            
            # Calculate delta
            delta = new_scope - current_scope
            
            # Acceptance criterion
            if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
                # Accept
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
            else:
                # Reject: rebuild the old row (should be same as before, but ensure consistency)
                # Actually, _row_rebuild already restores the old row if it fails,
                # but here it succeeded but we're rejecting, so we need to undo
                # This is tricky because _row_rebuild modified the global structures
                # Let's fix this: _row_rebuild should not modify global structures until accepted
                
                # For now, let's just revert by rebuilding the old row
                # This is inefficient but correct
                old_row = current_rows[row_idx]
                # Remove new row's diffs
                _remove_row_diffs(diff_to_rows, all_diffs, row_idx, new_row)
                # Add old row's diffs
                _add_row_diffs(diff_to_rows, all_diffs, row_idx, old_row)
                current_rows[row_idx] = old_row
                # current_scope remains unchanged
        
        # Cool down
        T *= cooling_rate
    
    return best_rows, best_scope

def _sa_vectorized(n_rows, k, initial_rows, time_limit, rng):
    """
    Vectorized SA using numpy for conflict counting.
    Move: Swap two rows.
    Objective: Minimize scope (max entry).
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    current_rows = np.array(initial_rows, dtype=np.int32)
    current_scope = _scope(current_rows.tolist())
    best_rows = current_rows.copy()
    best_scope = current_scope
    
    T = 5.0
    cooling_rate = 0.99995
    min_T = 0.01
    
    # Precompute differences for each row
    # diff_matrix[i, j] = current_rows[i][j+1] - current_rows[i][j] ? No, all pairwise diffs.
    # For k=5, row length is 6. Pairwise diffs are 15 per row.
    # Total diffs = n_rows * 15.
    
    def get_all_diffs(rows):
        diffs = []
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    diffs.append(r[j] - r[i])
        return np.array(diffs, dtype=np.int32)
    
    current_diffs = get_all_diffs(current_rows.tolist())
    
    # Count conflicts: number of diffs that appear more than once
    def count_conflicts(diffs):
        if len(diffs) == 0: return 0
        _, counts = np.unique(diffs, return_counts=True)
        # Conflicts are the sum of (count - 1) for all counts > 1
        # Or simply: if we want a valid set, we want 0 conflicts.
        # For SA, we can use the number of duplicate entries as a penalty.
        # Let's use the number of "extra" occurrences.
        return int(np.sum(counts[counts > 1] - 1))
    
    current_conflicts = count_conflicts(current_diffs)
    
    while time.time() < deadline and T > min_T:
        # Move: Swap two random rows
        if n_rows < 2:
            break
            
        idx1 = rng.randint(0, n_rows - 1)
        idx2 = rng.randint(0, n_rows - 1)
        if idx1 == idx2:
            continue
            
        # Perform swap
        current_rows[[idx1, idx2]] = current_rows[[idx2, idx1]]
        
        # Recalculate diffs and conflicts
        # This is O(n*k^2) which is fast for small n, k
        new_diffs = get_all_diffs(current_rows.tolist())
        new_conflicts = count_conflicts(new_diffs)
        
        new_scope = _scope(current_rows.tolist())
        
        # Objective: Primarily minimize conflicts (must be 0 for validity), secondarily minimize scope.
        # Let's define energy = conflicts * 1000 + scope
        old_energy = current_conflicts * 1000 + current_scope
        new_energy = new_conflicts * 1000 + new_scope
        
        delta = new_energy - old_energy
        
        if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
            # Accept
            current_conflicts = new_conflicts
            current_scope = new_scope
            if new_conflicts == 0 and new_scope < best_scope:
                best_scope = new_scope
                best_rows = current_rows.copy()
        else:
            # Reject: Swap back
            current_rows[[idx1, idx2]] = current_rows[[idx2, idx1]]
            
        T *= cooling_rate
        
    return best_rows.tolist(), best_scope

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, with all differences distinct across all rows.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    # Quick greedy baseline
    greedy_rows = _fast_greedy(n_rows, k)
    greedy_scope = _scope(greedy_rows)
    
    if not _valid(greedy_rows, k):
        return greedy_rows
    
    best_rows = greedy_rows
    best_scope = greedy_scope
    
    # Time budget
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed
    if remaining < 5:
        return best_rows
    
    # Phase 1: Vectorized SA with swaps to find a good structure
    # This is fast and can escape local minima of the greedy order
    sa_time_1 = min(remaining - 20, 60)
    if sa_time_1 > 5:
        sa_rows, sa_scope = _sa_vectorized(n_rows, k, best_rows, sa_time_1, rng)
        if sa_rows is not None and _valid(sa_rows, k):
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = sa_rows
    
    elapsed = time.time() - start_time
    remaining = _DEADLINE - elapsed
    
    # Phase 2: Row-rebuild SA to polish and compress
    # This is slower but more effective for final compression
    sa_time_2 = min(remaining - 10, 80)
    if sa_time_2 > 5:
        sa_rows, sa_scope = _sa_optimize(n_rows, k, best_rows, sa_time_2, rng)
        if sa_rows is not None and _valid(sa_rows, k):
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = sa_rows
    
    return best_rows
