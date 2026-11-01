import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def _row_diffs_set(row):
    """Return set of all pairwise differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _is_sidon(row):
    """Check if all pairwise differences in a row are distinct."""
    dl = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            dl.append(row[j] - row[i])
    return len(dl) == len(set(dl))

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)

def _build_greedy(n_rows, k):
    """Fast greedy: each row takes smallest next mark with no difference collision."""
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

def _backtrack_row(row_prefix, used, k, max_val, rng):
    """
    Try to complete a row starting from row_prefix (list of increasing ints starting at 0).
    Returns completed row or None.
    """
    if len(row_prefix) == k + 1:
        return row_prefix[:]
    
    last = row_prefix[-1]
    # Try candidate next values
    candidates = []
    for v in range(last + 1, max_val + 1):
        # Check Sidon within row
        new_diffs = {v - a for a in row_prefix}
        if len(new_diffs) != len(row_prefix):
            continue
        # Check no collision with used
        if new_diffs & used:
            continue
        candidates.append(v)
    
    if not candidates:
        return None
    
    # Shuffle for randomness
    rng.shuffle(candidates)
    
    for v in candidates:
        new_diffs = {v - a for a in row_prefix}
        new_row = row_prefix + [v]
        new_used = used | new_diffs
        result = _backtrack_row(new_row, new_used, k, max_val, rng)
        if result is not None:
            return result
    return None

def _try_construct_at_M(n_rows, k, M, rng, time_limit=5.0):
    """
    Try to construct n_rows rows of k+1 elements from {0,...,M} with all diffs distinct.
    Uses backtracking with randomization.
    Returns list of rows or None.
    """
    start = time.time()
    used = set()
    rows = []
    
    for ri in range(n_rows):
        if time.time() - start > time_limit:
            return None
        
        # Try multiple random orderings for this row
        best_row = None
        attempts = 20
        for attempt in range(attempts):
            row = [0]
            # Use randomized backtracking
            result = _backtrack_row(row, used, k, M, rng)
            if result is not None:
                best_row = result
                break
        
        if best_row is None:
            return None
        
        rows.append(best_row)
        used |= _row_diffs_set(best_row)
    
    return rows

def _resolve_row_given_others(rows, ri, k, M, rng, used_excl_ri):
    """
    Try to re-solve row ri given other rows are fixed.
    used_excl_ri is the set of all diffs from rows other than ri.
    Returns new row or None.
    """
    attempts = 30
    for attempt in range(attempts):
        row = [0]
        result = _backtrack_row(row, used_excl_ri, k, M, rng)
        if result is not None:
            return result
    return None

def construct(n_rows, k):
    """
    Construct a difference triangle set for n_rows rows of k+1 integers.
    Returns a list of rows.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    rng = random.Random(42)
    
    # Start from greedy
    best = _build_greedy(n_rows, k)
    best_val = _max_entry(best)
    
    # Counting bound lower bound
    # Total diffs needed: n_rows * k*(k+1)/2
    # Available diffs in {1,...,M}: M
    # So M >= n_rows * k*(k+1)/2
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed
    
    # Try to find solution at various M values starting from lb
    # First try greedy value, then try lower values
    current_M = best_val
    
    # Phase 1: Try to improve by solving at lower M values
    for target_M in range(max(lb, current_M - 50), current_M):
        if time.time() > deadline:
            break
        
        # Multiple restarts
        found = False
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            result = _try_construct_at_M(n_rows, k, target_M, rng, time_limit=3.0)
            if result is not None:
                if _valid(result, n_rows, k):
                    best = result
                    best_val = _max_entry(best)
                    found = True
                    break
        
        if found:
            current_M = best_val
            continue
    
    # Phase 2: Local search with row-level re-solving to further reduce max
    cur = [r[:] for r in best]
    cur_val = best_val
    
    # Build used set
    cur_used = set()
    for r in cur:
        cur_used |= _row_diffs_set(r)
    
    no_improve = 0
    max_no_improve = 30000
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random row to re-solve
        ri = rng.randrange(n_rows)
        
        # Compute used excluding row ri
        row_ri_diffs = _row_diffs_set(cur[ri])
        used_excl = cur_used - row_ri_diffs
        
        # Try to re-solve row ri with a tighter max constraint
        # Target: reduce the max entry of this row
        old_max = max(cur[ri])
        target_max = old_max - 1
        
        new_row = None
        attempts = 50
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            # Try to find a row with max <= target_max
            result = _backtrack_row([0], used_excl, k, target_max, rng)
            if result is not None:
                new_row = result
                break
        
        if new_row is not None:
            new_max_entry = max(max(r) for r in cur if r is not cur[ri])
            new_max_entry = max(new_max_entry, max(new_row))
            
            if new_max_entry < cur_val:
                # Accept improvement
                cur_used -= row_ri_diffs
                cur_used |= _row_diffs_set(new_row)
                cur[ri] = new_row
                cur_val = new_max_entry
                best = [r[:] for r in cur]
                best_val = cur_val
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # Also try swapping two rows to break symmetry
        if no_improve % 100 == 0 and n_rows >= 2:
            ri1 = rng.randrange(n_rows)
            ri2 = rng.randrange(n_rows)
            if ri1 != ri2:
                cur[ri1], cur[ri2] = cur[ri2], cur[ri1]
                # This doesn't change validity or max, but may help escape local optima
                # Actually it doesn't help for max, so revert
                cur[ri1], cur[ri2] = cur[ri2], cur[ri1]
    
    # Ensure best is valid
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    
    return best

# EVOLVE-BLOCK-END
