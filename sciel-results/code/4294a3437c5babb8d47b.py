import sys
import time
import random

sys.setrecursionlimit(10000)


def _check_valid(rows, k):
    """Check if the set of rows is valid (all differences distinct across all rows)."""
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


def _max_entry(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, used_diffs=None, max_val=None):
    """Greedy construction: build each row by picking the smallest valid value at each position."""
    if used_diffs is None:
        used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if max_val is not None and x > max_val:
                return None
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
                used_diffs |= new_diffs
        rows.append(row)
    return rows


def _greedy_variant(n_rows, k, randomize=False, max_val=None):
    """Greedy with optional randomization in value selection."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if max_val is not None and x > max_val:
                return None
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
                used_diffs |= new_diffs
            elif randomize and random.random() < 0.1:
                # Skip a few values randomly to diversify
                x += random.randint(1, 3)
                if max_val is not None and x > max_val:
                    return None
                continue
    return rows


def _build_used_set(rows):
    """Build the set of all differences used across all rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    return used


def _get_diffs_for_pos(row, p, val):
    """Get the set of differences created by inserting val at position p."""
    diffs = set()
    for i in range(p):
        diffs.add(val - row[i])
    return diffs


def _can_set(row, p, val, used_diffs):
    """Check if we can set row[p] = val without conflicts."""
    if val <= row[p - 1]:
        return False
    diffs = _get_diffs_for_pos(row, p, val)
    return not diffs & used_diffs


def _apply_set(row, p, val, used_diffs):
    """Apply setting row[p] = val, update used_diffs, return diffs added."""
    diffs = _get_diffs_for_pos(row, p, val)
    row[p] = val
    used_diffs |= diffs
    return diffs


def _fast_compact_row(row, used_diffs, k):
    """Try to compact a row by shifting values down where possible."""
    for p in range(1, k + 1):
        # Try to reduce row[p] as much as possible
        old_val = row[p]
        for new_val in range(row[p - 1] + 1, old_val):
            if _can_set(row, p, new_val, used_diffs):
                diffs = _get_diffs_for_pos(row, p, new_val)
                # Remove old diffs
                for i in range(p):
                    used_diffs.discard(old_val - row[i])
                # Set new value
                row[p] = new_val
                used_diffs |= diffs
                break
    return row


def _local_compact(rows, k):
    """Apply local compaction to all rows."""
    used_diffs = _build_used_set(rows)
    for r in range(len(rows)):
        _fast_compact_row(rows[r], used_diffs, k)
    return rows


def _rebuild_row(rows, r, k, used_diffs, randomize=False):
    """Rebuild row r from scratch given the used_diffs from other rows."""
    # Remove row r's diffs from used_diffs
    row = rows[r]
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            used_diffs.discard(row[j] - row[i])
    
    # Rebuild
    new_row = [0]
    x = 0
    while len(new_row) < k + 1:
        x += 1
        if x > 200:  # Safety cap
            break
        new_diffs = set()
        valid = True
        for a in new_row:
            d = x - a
            if d in used_diffs:
                valid = False
                break
            new_diffs.add(d)
        if valid:
            new_row.append(x)
            used_diffs |= new_diffs
        elif randomize and random.random() < 0.15:
            x += random.randint(1, 4)
    
    if len(new_row) == k + 1:
        rows[r] = new_row
        return True
    else:
        # Restore old row
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
        return False


def _simulated_annealing(n_rows, k, initial_rows, time_limit=100.0):
    """Simulated annealing with row-level perturbation and local compaction."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    current_rows = [row[:] for row in initial_rows]
    current_scope = _max_entry(current_rows)
    best_rows = [row[:] for row in current_rows]
    best_scope = current_scope
    
    T = 50.0  # Initial temperature
    cooling = 0.9995
    
    used_diffs = _build_used_set(current_rows)
    
    while time.time() < deadline and T > 0.1:
        # Pick a random row to perturb
        r = random.randint(0, n_rows - 1)
        
        # Save current state
        old_row = current_rows[r][:]
        old_scope = current_scope
        
        # Rebuild the row with some randomization
        success = _rebuild_row(current_rows, r, k, used_diffs, randomize=True)
        
        if not success:
            T *= cooling
            continue
        
        # Apply local compaction to the modified row
        _fast_compact_row(current_rows[r], used_diffs, k)
        
        new_scope = _max_entry(current_rows)
        
        # Acceptance criterion
        if new_scope <= old_scope or random.random() < pow(2.718281828, (old_scope - new_scope) / T):
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [row[:] for row in current_rows]
        else:
            # Revert
            current_rows[r] = old_row
            current_scope = old_scope
            # Rebuild used_diffs
            used_diffs = _build_used_set(current_rows)
        
        T *= cooling
        
        # Periodic full compaction
        if random.random() < 0.05:
            used_diffs = _build_used_set(current_rows)
            for rr in range(n_rows):
                _fast_compact_row(current_rows[rr], used_diffs, k)
            new_scope = _max_entry(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [row[:] for row in current_rows]
            current_scope = new_scope
    
    return best_rows, best_scope


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Total differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    # Phase 1: Generate multiple greedy solutions
    best_rows = None
    best_scope = float('inf')
    
    # Deterministic greedy
    rows_greedy = _greedy_construct(n_rows, k)
    if rows_greedy is not None:
        scope = _max_entry(rows_greedy)
        if scope < best_scope:
            best_scope = scope
            best_rows = [row[:] for row in rows_greedy]
    
    # Randomized greedy variants
    for trial in range(20):
        if time.time() > deadline:
            break
        random.seed(trial * 7919 + 13)
        rows = _greedy_variant(n_rows, k, randomize=True)
        if rows is not None:
            scope = _max_entry(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [row[:] for row in rows]
    
    if best_rows is None:
        # Fallback
        rows_greedy = _greedy_construct(n_rows, k)
        return rows_greedy if rows_greedy else [[0] * (k + 1) for _ in range(n_rows)]
    
    # Phase 2: Local compaction on best solution
    best_rows = _local_compact(best_rows, k)
    best_scope = _max_entry(best_rows)
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 10.0:
        sa_rows, sa_scope = _simulated_annealing(n_rows, k, best_rows, time_limit=min(80.0, deadline - time.time() - 5.0))
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows
    
    # Phase 4: Final compaction
    best_rows = _local_compact(best_rows, k)
    best_scope = _max_entry(best_rows)
    
    # Verify validity
    if not _check_valid(best_rows, k):
        # Fallback to greedy
        used = set()
        rows_greedy = []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows_greedy.append(row)
        return rows_greedy
    
    return best_rows
