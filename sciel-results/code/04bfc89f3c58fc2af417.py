import time
import random
import sys

def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows, each of k+1 integers
    starting at 0, strictly increasing, with all positive differences distinct
    across all rows. Minimize the largest entry.
    """
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    M_lower = int((2 * total_diffs_needed) ** 0.5)
    while M_lower * (M_lower + 1) // 2 < total_diffs_needed:
        M_lower += 1
    
    deadline = time.time() + 115.0
    best_rows = None
    best_M = None
    
    # Initial greedy construction
    rows = _greedy_construct(n_rows, k)
    if rows is not None:
        best_M = max(row[-1] for row in rows)
        best_rows = rows
    
    # Try simulated annealing to reduce the scope
    while time.time() < deadline - 5.0:
        # Pick starting M: try to go below current best
        start_M = best_M - 1 if best_M > M_lower else best_M
        if start_M < M_lower:
            start_M = M_lower
        
        # Multiple restarts at this M level
        for restart in range(3):
            if time.time() > deadline - 5.0:
                break
            
            # Build initial solution at this M
            init_rows = _build_initial(rows, n_rows, k, start_M)
            if init_rows is None:
                continue
            
            # Run SA
            sa_rows, sa_M = _sa_run(init_rows, n_rows, k, start_M, deadline - 5.0)
            if sa_rows is not None:
                if best_M is None or sa_M < best_M:
                    best_M = sa_M
                    best_rows = sa_rows
                    if best_M == M_lower:
                        return best_rows
        
        # If we couldn't improve, try one more level down
        if best_M > M_lower and time.time() < deadline - 10.0:
            # Try direct construction at lower M
            for attempt in range(5):
                if time.time() > deadline - 5.0:
                    break
                trial = _build_initial(None, n_rows, k, M_lower)
                if trial is not None:
                    sa_rows, sa_M = _sa_run(trial, n_rows, k, M_lower, deadline - 5.0)
                    if sa_rows is not None and (best_M is None or sa_M < best_M):
                        best_M = sa_M
                        best_rows = sa_rows
                        if best_M == M_lower:
                            return best_rows
    
    return best_rows if best_rows is not None else _greedy_construct(n_rows, k)


def _greedy_construct(n_rows, k):
    """
    Greedy construction: for each row, pick the smallest possible next value
    that doesn't conflict with existing differences.
    """
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


def _build_initial(existing_rows, n_rows, k, M):
    """
    Build an initial valid set of rows with max entry <= M.
    If existing_rows provided, try to adapt them; otherwise build fresh.
    """
    used_diffs = set()
    rows = []
    
    for r in range(n_rows):
        if existing_rows is not None and r < len(existing_rows):
            # Try to use existing row if it fits
            row = existing_rows[r]
            if row[-1] <= M:
                # Check all diffs are available
                row_diffs = set()
                valid = True
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d in used_diffs or d in row_diffs:
                            valid = False
                            break
                        row_diffs.add(d)
                    if not valid:
                        break
                if valid:
                    rows.append(list(row))
                    used_diffs |= row_diffs
                    continue
        
        # Build a new row
        row = _find_row_greedy(used_diffs, M, k)
        if row is None:
            return None
        rows.append(row)
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    return rows


def _find_row_greedy(used_diffs, M, k):
    """Find a row greedily with some randomization."""
    row = [0]
    current_diffs = set()
    
    for pos in range(1, k + 1):
        best_x = None
        best_score = float('inf')
        
        # Try candidates from low to high, with some randomization
        candidates = list(range(row[-1] + 1, M + 1))
        random.shuffle(candidates)
        # But prefer smaller values
        candidates.sort()
        
        found = False
        for x in candidates:
            valid = True
            new_diffs = set()
            for a in row:
                d = x - a
                if d in used_diffs or d in current_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                best_x = x
                found = True
                break
        
        if not found:
            return None
        
        row.append(best_x)
        current_diffs.update({best_x - a for a in row[:-1]})
    
    return row


def _row_diffs_set(row):
    """Compute set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _all_diffs_valid(rows):
    """Check if all differences across all rows are distinct."""
    seen = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _max_entry(rows):
    return max(row[-1] for row in rows)


def _sa_run(rows, n_rows, k, M_init, deadline):
    """
    Simulated annealing to minimize the largest entry.
    Uses multiple move types: position shift, row re-solve, row swap.
    """
    if rows is None:
        return None, None
    
    current = [list(r) for r in rows]
    current_M = _max_entry(current)
    best = [list(r) for r in current]
    best_M = current_M
    
    # Precompute used diffs for current state
    used_diffs = set()
    for row in current:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    # SA parameters
    T = 50.0  # Initial temperature
    cooling = 0.995
    min_T = 0.1
    steps_per_temp = 200
    
    import math
    
    while T > min_T:
        for step in range(steps_per_temp):
            if time.time() > deadline:
                break
            
            # Choose move type
            move_type = random.random()
            
            if move_type < 0.5:
                # Single position shift
                new_state, delta = _sa_move_single_pos(current, used_diffs, k)
            elif move_type < 0.8:
                # Re-solve one row
                new_state, delta = _sa_move_resolve_row(current, used_diffs, n_rows, k)
            else:
                # Row swap (try swapping order to allow different greedy fills)
                new_state, delta = _sa_move_row_swap(current, used_diffs, n_rows, k)
            
            if new_state is None:
                continue
            
            new_M = _max_entry(new_state)
            
            # Accept if better or with probability exp(-delta/T)
            if new_M < best_M:
                best = [list(r) for r in new_state]
                best_M = new_M
            
            if delta <= 0:
                # Accept improvement
                current = new_state
                current_M = new_M
            else:
                # Accept worsening with probability
                if random.random() < math.exp(-delta / T):
                    current = new_state
                    current_M = new_M
        
        T *= cooling
    
    return best, best_M


def _sa_move_single_pos(rows, used_diffs, k):
    """
    Try shifting one position in one row.
    Returns (new_rows, delta) where delta = new_max - old_max (approximate).
    """
    n_rows = len(rows)
    r = random.randint(0, n_rows - 1)
    pos = random.randint(1, len(rows[r]) - 1)  # Don't move position 0
    
    row = list(rows[r])
    old_val = row[pos]
    
    # Compute diffs involving position pos
    old_diffs = set()
    for i in range(len(row)):
        if i != pos:
            d = abs(row[pos] - row[i])
            old_diffs.add(d)
    
    # Try new values
    min_allowed = row[pos - 1] + 1
    max_allowed = row[pos + 1] - 1 if pos + 1 < len(row) else 10000
    
    # Also need to respect that new diffs don't conflict with used_diffs minus old_diffs
    other_diffs = used_diffs - old_diffs
    
    candidates = list(range(min_allowed, max_allowed + 1))
    random.shuffle(candidates)
    candidates.sort(key=lambda x: abs(x - old_val))
    
    for new_val in candidates:
        # Check new diffs
        new_diffs = set()
        valid = True
        for i in range(len(row)):
            if i != pos:
                d = abs(new_val - row[i])
                if d in other_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
        
        if valid:
            row[pos] = new_val
            new_rows = [list(rows[i]) for i in range(len(rows))]
            new_rows[r] = row
            
            new_M = _max_entry(new_rows)
            old_M = _max_entry(rows)
            delta = new_M - old_M
            
            return new_rows, delta
    
    return None, 0


def _sa_move_resolve_row(rows, used_diffs, n_rows, k):
    """
    Re-solve one row from scratch, given the other rows' diffs are fixed.
    """
    r = random.randint(0, n_rows - 1)
    
    # Compute diffs from all other rows
    other_diffs = set()
    for i in range(n_rows):
        if i == r:
            continue
        for a in range(len(rows[i])):
            for b in range(a + 1, len(rows[i])):
                other_diffs.add(rows[i][b] - rows[i][a])
    
    # Remove current row's diffs from used_diffs to get other_diffs
    # Actually let's just recompute
    other_diffs = set()
    for i in range(n_rows):
        if i == r:
            continue
        row = rows[i]
        for a in range(len(row)):
            for b in range(a + 1, len(row)):
                other_diffs.add(row[b] - row[a])
    
    # Find a new row that avoids other_diffs
    new_row = _find_row_greedy(other_diffs, 10000, k)
    if new_row is None:
        return None, 0
    
    new_rows = [list(rows[i]) for i in range(len(rows))]
    new_rows[r] = new_row
    
    new_M = _max_entry(new_rows)
    old_M = _max_entry(rows)
    delta = new_M - old_M
    
    return new_rows, delta


def _sa_move_row_swap(rows, used_diffs, n_rows, k):
    """
    Swap two rows to potentially allow different subsequent fills.
    This is a heuristic move; the actual benefit comes from re-solving.
    """
    if n_rows < 2:
        return None, 0
    
    i, j = random.sample(range(n_rows), 2)
    new_rows = [list(rows[r]) for r in range(n_rows)]
    new_rows[i], new_rows[j] = new_rows[j], new_rows[i]
    
    # Validate
    if not _all_diffs_valid(new_rows):
        return None, 0
    
    new_M = _max_entry(new_rows)
    old_M = _max_entry(rows)
    delta = new_M - old_M
    
    return new_rows, delta
