"""Difference triangle set (4,7): minimize largest entry."""
import time
import random
import math

# Deadline guard
_DEADLINE = 145.0  # seconds wall-clock limit for evaluator (5s safety margin)
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _row_diffs_set(row):
    """Return set of all positive differences in a row."""
    n = len(row)
    return {row[j] - row[i] for i in range(n) for j in range(i + 1, n)}

def _is_valid(rows):
    """Check that all positive differences are distinct across all rows."""
    seen = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                d = r[j] - r[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _score(rows):
    """The largest entry across all rows (lower is better)."""
    return max(max(r) for r in rows)

def _greedy_row(used, k, start=1, jitter=0.0, window=50):
    """
    Build one row of k+1 entries starting at 0, given used diffs.
    Uses a randomized greedy approach with look-ahead and optional jitter.
    """
    row = [0]
    x = start
    current_row_diffs = set()
    
    while len(row) < k + 1:
        found = False
        # Look ahead window
        for trial_x in range(x + 1, x + window + 1):
            new_diffs = {trial_x - a for a in row}
            if not (new_diffs & used) and not (new_diffs & current_row_diffs):
                # Jitter: skip valid candidates to explore different structures
                if jitter > 0 and random.random() < jitter:
                    continue
                row.append(trial_x)
                current_row_diffs |= new_diffs
                found = True
                break
        
        if not found:
            # Fallback: expand window significantly
            # We need to find *some* valid x, so we scan further
            # To keep it fast, we limit the fallback scan
            for trial_x in range(x + 1, x + 500):
                new_diffs = {trial_x - a for a in row}
                if not (new_diffs & used) and not (new_diffs & current_row_diffs):
                    row.append(trial_x)
                    current_row_diffs |= new_diffs
                    found = True
                    break
        
        if not found:
            # If we still haven't found one, we might be stuck in a bad configuration
            # Return a dummy or raise? For robustness, we return a row that is likely invalid
            # but the caller should check validity.
            # Actually, for n_rows=4, k=7, scope ~134, greedy with window 500 should always work.
            # If it fails, it means the 'used' set is extremely dense or we hit a dead end.
            # We return a row with large gaps to ensure it's distinct, though it might be suboptimal.
            # Better: return None to signal failure.
            return None
        
        x = row[-1]
    
    return row

def _construct_rows(n_rows, k, seed=0):
    """Construct n_rows rows using randomized greedy."""
    random.seed(seed)
    used = set()
    rows = []
    
    for _ in range(n_rows):
        # Start with some jitter to explore
        jitter = random.uniform(0.0, 0.1)
        window = 50 + random.randint(0, 20)
        
        row = _greedy_row(used, k, start=1, jitter=jitter, window=window)
        
        if row is None:
            # Fallback to strict greedy if randomized fails
            row = _greedy_row(used, k, start=1, jitter=0.0, window=500)
            if row is None:
                # This should never happen for small n,k
                return None
        
        # Update used diffs
        used |= _row_diffs_set(row)
        rows.append(row)
        
    return rows

def _optimize_row(rows, row_idx, k, max_attempts=20):
    """
    Try to find a better (smaller max entry) version of rows[row_idx]
    while keeping other rows fixed.
    Returns the best row found.
    """
    current_row = rows[row_idx]
    current_max = max(current_row)
    best_row = list(current_row)
    best_max = current_max
    
    # Calculate diffs used by other rows
    used_other = set()
    for j in range(len(rows)):
        if j != row_idx:
            used_other |= _row_diffs_set(rows[j])
            
    for attempt in range(max_attempts):
        # Increase jitter and window slightly as we try more
        jitter = 0.05 + 0.05 * (attempt / max_attempts)
        window = 40 + 20 * (attempt / max_attempts)
        
        new_row = _greedy_row(used_other, k, start=1, jitter=jitter, window=int(window))
        
        if new_row is not None:
            new_max = max(new_row)
            if new_max < best_max:
                # Verify validity with the rest of the rows (should be true by construction
                # since we avoided used_other and internal diffs are unique in greedy)
                # But just to be safe, we assume it's valid relative to others.
                best_row = new_row
                best_max = new_max
                
    return best_row

def _local_search(rows, n_rows, k, time_limit=10.0):
    """
    Iteratively improve rows by optimizing each one individually.
    """
    rows = [list(r) for r in rows]
    start_time = time.time()
    
    improved = True
    while improved and time.time() - start_time < time_limit and _time_left() > 5.0:
        improved = False
        # Process rows in order of largest max entry first
        order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
        
        for i in order:
            if time.time() - start_time > time_limit or _time_left() < 5.0:
                break
                
            new_row = _optimize_row(rows, i, k, max_attempts=10)
            if new_row is not None and max(new_row) < max(rows[i]):
                rows[i] = new_row
                improved = True
                
    return rows

def _kick(rows, n_rows, k):
    """
    Perturb the solution to escape local minima.
    Rebuilds 1 or 2 random rows with high randomness.
    """
    rows = [list(r) for r in rows]
    num_kicks = random.choice([1, 2])
    indices = random.sample(range(n_rows), num_kicks)
    
    for idx in indices:
        # High jitter to significantly change the row
        jitter = random.uniform(0.2, 0.4)
        window = 30 + random.randint(0, 30)
        
        # We need to be careful: if we rebuild a row, it might conflict with others.
        # The simplest kick is to just replace the row with a new greedy one
        # that avoids the diffs of *all other* rows.
        used_other = set()
        for j in range(n_rows):
            if j != idx:
                used_other |= _row_diffs_set(rows[j])
                
        new_row = _greedy_row(used_other, k, start=1, jitter=jitter, window=window)
        if new_row is not None:
            rows[idx] = new_row
        else:
            # If it fails, keep the old row
            pass
            
    return rows

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry subject to all positive differences being distinct."""
    global _START
    _START = time.time()
    
    best_rows = None
    best_score = float('inf')
    
    # Initial construction
    initial_rows = _construct_rows(n_rows, k, seed=42)
    if initial_rows and _is_valid(initial_rows):
        best_rows = initial_rows
        best_score = _score(best_rows)
    else:
        # Fallback if initial fails (should not happen)
        best_rows = [[0, 1, 2, 3, 4, 5, 6, 7] for _ in range(n_rows)] # Dummy
        best_score = _score(best_rows)

    # ILS Loop
    current_rows = [list(r) for r in best_rows]
    no_improvement_count = 0
    max_no_improvement = 5
    
    while _time_left() > 10.0:
        # Local Search
        ls_rows = _local_search(current_rows, n_rows, k, time_limit=min(5.0, _time_left() - 10.0))
        
        if _is_valid(ls_rows):
            ls_score = _score(ls_rows)
            if ls_score < best_score:
                best_rows = ls_rows
                best_score = ls_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            current_rows = ls_rows
        else:
            # If local search broke validity (unlikely with our row optimizer), revert
            current_rows = [list(r) for r in best_rows]
            no_improvement_count += 1
            
        if no_improvement_count >= max_no_improvement:
            # Kick to escape local minimum
            if _time_left() < 15.0:
                break
            current_rows = _kick(current_rows, n_rows, k)
            no_improvement_count = 0
            # After a kick, we might have a worse score, but we expect local search to fix it
            # If the kick resulted in an invalid state (shouldn't happen with our kick), 
            # we revert to best.
            if not _is_valid(current_rows):
                current_rows = [list(r) for r in best_rows]

    # Final polish on the best solution found
    if _time_left() > 5.0:
        final_rows = _local_search(best_rows, n_rows, k, time_limit=_time_left() - 5.0)
        if _is_valid(final_rows) and _score(final_rows) < best_score:
            best_rows = final_rows
            best_score = _score(best_rows)

    return best_rows

if __name__ == "__main__":
    # Quick self-test
    rows = construct(4, 7)
    print(f"Score: {_score(rows)}")
    print(f"Valid: {_is_valid(rows)}")
    print(rows)
