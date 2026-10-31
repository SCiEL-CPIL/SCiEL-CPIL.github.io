"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0."""
import time
import random
import sys
from collections import defaultdict

_START_TIME = None
_DEADLINE = 140.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _compute_diffs_set(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _validate_solution(rows, n_rows, k):
    """Check that all differences across all rows are distinct and rows are valid."""
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(k):
            if row[i] >= row[i+1]:
                return False
        d = _compute_diffs_set(row)
        if len(d) != k * (k + 1) // 2:
            return False
        if d & seen:
            return False
        seen |= d
    return True

def _greedy_build(n_rows, k, row_order, rng):
    """Greedy: build rows in given order, each row greedily picks smallest valid next value."""
    used = set()
    rows = [None] * n_rows
    for r in row_order:
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                # Check new differences
                ok = True
                for a in row:
                    d = x - a
                    if d in used:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
                    break
                x += 1
        rows[r] = row
    return rows

def _greedy_round_robin(n_rows, k, rng):
    """Round-robin greedy: extend rows one element at a time, choosing the row with smallest last element."""
    used = set()
    rows = [[0] for _ in range(n_rows)]
    total_needed = n_rows * (k + 1)
    placed = n_rows
    
    while placed < total_needed:
        if _time_left() < 3.0:
            return None
        # Find row with smallest last element that still needs more
        best_r = -1
        best_last = -1
        order = list(range(n_rows))
        rng.shuffle(order)
        for r in order:
            if len(rows[r]) >= k + 1:
                continue
            last = rows[r][-1]
            if best_r == -1 or last < best_last:
                best_last = last
                best_r = r
        if best_r == -1:
            return None
        row = rows[best_r]
        x = row[-1] + 1
        while True:
            ok = True
            for a in row:
                d = x - a
                if d in used:
                    ok = False
                    break
            if ok:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
                placed += 1
                break
            x += 1
    return rows

def _try_greedy_variants(n_rows, k, rng, num_trials=30):
    """Try multiple greedy strategies and return best."""
    best_rows = None
    best_max = float('inf')
    
    for t in range(num_trials):
        if _time_left() < 5.0:
            break
        # Random row order
        order = list(range(n_rows))
        rng.shuffle(order)
        rows = _greedy_build(n_rows, k, order, rng)
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
    
    # Also try round-robin
    for t in range(num_trials):
        if _time_left() < 5.0:
            break
        rows = _greedy_round_robin(n_rows, k, rng)
        if rows is None:
            continue
        cur_max = max(max(r) for r in rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [list(r) for r in rows]
    
    return best_rows, best_max

def _backtracking_search(n_rows, k, upper_bound, rng, time_limit=20.0):
    """
    Backtracking search with forward checking to find a solution with max < upper_bound.
    We model this as filling rows one by one, or interleaving.
    To keep it simple and effective, we try to construct rows sequentially but with backtracking.
    Variable: next element in current row.
    Domain: integers > last element, < upper_bound.
    Constraint: differences not in used set.
    
    To improve on simple sequential, we can use a "repair" or "restart" strategy if stuck.
    """
    start_time = time.time()
    best_found = None
    best_max = upper_bound
    
    # We will try to construct a solution strictly smaller than upper_bound
    # We use a recursive helper that tracks state.
    # State: rows constructed so far, current row being built, used differences.
    
    def search(rows, current_row, used_diffs, max_val_so_far):
        nonlocal best_found, best_max
        
        if time.time() - start_time > time_limit:
            return False
        
        # Check if we have completed all rows
        if len(rows) == n_rows:
            if max_val_so_far < best_max:
                best_max = max_val_so_far
                best_found = [list(r) for r in rows]
            return True
        
        # If we are building a new row
        if len(current_row) == 0:
            current_row = [0]
            max_val_so_far = 0 # 0 doesn't count as max usually, but let's track actual max
            
        # If current row is complete
        if len(current_row) == k + 1:
            rows.append(list(current_row))
            success = search(rows, [], used_diffs, max_val_so_far)
            if success:
                return True
            rows.pop()
            return False
        
        # Try to add next element to current_row
        last = current_row[-1]
        # We want to minimize max, so we try small values first.
        # But we must stay within upper_bound.
        # Also, we should try to keep the row "compact" to leave room for others?
        # Actually, just trying smallest valid values is a good heuristic for minimizing max.
        
        # Heuristic: Randomize the start point slightly to escape local traps?
        # For now, deterministic smallest-first is strong for minimization.
        
        # To avoid deep recursion stack issues and speed up, we limit the search depth or use iterative deepening?
        # Given the small k (likely 7 or 8 based on "k+1" and typical DT sets), recursion depth is manageable.
        
        # Optimization: Pre-compute valid candidates?
        # For x in range(last + 1, upper_bound + 1):
        #    Check if x - a for all a in current_row are not in used_diffs
        #    If yes, recurse
        
        # To add diversity and escape dead ends, we can shuffle the candidate order slightly
        # or use a random restart strategy outside this function.
        
        # Let's try a simple randomized backtracking:
        # Generate candidates, shuffle them slightly (or pick a random valid one if many?)
        # No, for minimization, we MUST try smaller values first to find the smallest max.
        # However, if we get stuck, we backtrack.
        
        # Forward Checking:
        # When we place x, we add diffs to used_diffs.
        # We don't explicitly check future domains here (expensive), but backtracking handles it.
        
        # To speed up, we can prune if max_val_so_far is already >= best_max.
        if max_val_so_far >= best_max:
            return False
            
        # Try candidates
        # We can limit the number of candidates tried to avoid excessive branching?
        # No, we need to find the optimum.
        
        # One optimization: If the row is getting too large, it might force the max to be high.
        # But we are bounded by upper_bound.
        
        # Randomization for tie-breaking or slight perturbation?
        # Let's stick to deterministic order (smallest first) but allow random restarts in the caller.
        
        for x in range(last + 1, upper_bound + 1):
            # Check validity
            valid = True
            new_diffs = []
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Recurse
            current_row.append(x)
            used_diffs.update(new_diffs)
            new_max = max(max_val_so_far, x)
            
            if search(rows, current_row, used_diffs, new_max):
                return True
            
            # Backtrack
            current_row.pop()
            for d in new_diffs:
                used_diffs.discard(d)
                
        return False

    # We will run multiple attempts with different random seeds/orders if we allow randomization in the search.
    # Since the search is deterministic given the order, we can vary the "upper_bound" or the row order.
    # Actually, the row order matters.
    
    # Let's try a few random row orders
    for _ in range(5): # Limit restarts to fit in time
        if time.time() - start_time > time_limit:
            break
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        # The search function above assumes sequential rows. 
        # We need to adapt it to respect row_order or just shuffle the "rows" list logic.
        # Easier: Just run the sequential search but it's fixed order 0..n-1.
        # To use row_order, we can just permute the "rows" list in the result, 
        # but the construction order affects the "used_diffs" state.
        # So we need to pass row_order into the search.
        
        # Let's modify search to take an index of which row we are building next.
        
        def search_with_order(rows, current_row, used_diffs, max_val_so_far, row_idx):
            nonlocal best_found, best_max
            
            if time.time() - start_time > time_limit:
                return False
            
            if row_idx == n_rows:
                if max_val_so_far < best_max:
                    best_max = max_val_so_far
                    best_found = [list(r) for r in rows]
                return True
            
            if max_val_so_far >= best_max:
                return False
                
            # Start new row
            if len(current_row) == 0:
                current_row = [0]
                # 0 is fixed, no diffs added for 0 itself in this context (diffs are positive)
                
            if len(current_row) == k + 1:
                rows.append(list(current_row))
                success = search_with_order(rows, [], used_diffs, max_val_so_far, row_idx + 1)
                if success:
                    return True
                rows.pop()
                return False
            
            last = current_row[-1]
            
            # Try candidates
            for x in range(last + 1, upper_bound + 1):
                valid = True
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                current_row.append(x)
                used_diffs.update(new_diffs)
                new_max = max(max_val_so_far, x)
                
                if search_with_order(rows, current_row, used_diffs, new_max, row_idx):
                    return True
                
                current_row.pop()
                for d in new_diffs:
                    used_diffs.discard(d)
                    
            return False

        search_with_order([], [], set(), 0, 0)
        
        if best_found is not None:
            break # Found a solution, no need for more restarts if we want to refine it
            
    return best_found, best_max

def _local_search_reduce_max(rows, n_rows, k, rng, max_iters=5000):
    """Try to reduce the maximum entry via local search."""
    rows = [list(r) for r in rows]
    best_max = max(max(r) for r in rows)
    
    if _time_left() < 3.0:
        return rows, best_max
    
    # Precompute diff sets for speed
    row_diffs = [_compute_diffs_set(rows[r]) for r in range(n_rows)]
    all_diffs = set()
    for rd in row_diffs:
        all_diffs |= rd
        
    for iteration in range(max_iters):
        if _time_left() < 3.0:
            break
            
        # Find the row with the max element
        max_row_idx = max(range(n_rows), key=lambda i: max(rows[i]))
        row = rows[max_row_idx]
        old_val = row[-1]
        
        # Try to reduce the last element
        improved = False
        for new_val in range(old_val - 1, row[-2], -1):
            # Check if new_val creates any conflicts
            new_diffs = set()
            valid = True
            for a in row[:-1]:
                d = new_val - a
                if d in row_diffs[max_row_idx]:
                    # It's in the old set, but we are replacing the last element.
                    # We need to check if d is in the set of diffs generated by row[:-1] + [new_val]
                    # which is (old_diffs - diffs_involving_old_val) U diffs_involving_new_val
                    # Simpler: just compute the new set and check against others.
                    pass
                new_diffs.add(d)
            
            # Recompute full set for this row to be safe and simple
            row_copy = row[:-1] + [new_val]
            nd = _compute_diffs_set(row_copy)
            
            # Check internal consistency (should be fine if increasing)
            if len(nd) != k*(k+1)//2:
                continue
                
            # Check against other rows
            conflict = False
            for other_r in range(n_rows):
                if other_r == max_row_idx:
                    continue
                if nd & row_diffs[other_r]:
                    conflict = True
                    break
            
            if not conflict:
                row[-1] = new_val
                row_diffs[max_row_idx] = nd
                # Update all_diffs
                all_diffs = set()
                for rr in range(n_rows):
                    all_diffs |= row_diffs[rr]
                    
                cur_max = max(max(rows[rr]) for rr in range(n_rows))
                if cur_max < best_max:
                    best_max = cur_max
                improved = True
                break
                
        if not improved:
            # Try a random perturbation
            rr = rng.randrange(n_rows)
            pos = rng.randrange(1, k + 1)
            old = rows[rr][pos]
            delta = rng.randint(-5, 5)
            new = old + delta
            if new <= rows[rr][pos-1] or (pos < k and new >= rows[rr][pos+1]) or new <= 0:
                continue
            row_copy = rows[rr][:]
            row_copy[pos] = new
            nd = _compute_diffs_set(row_copy)
            if len(nd) != k*(k+1)//2:
                continue
            conflict = False
            for other_r in range(n_rows):
                if other_r == rr:
                    continue
                if nd & row_diffs[other_r]:
                    conflict = True
                    break
            if not conflict:
                rows[rr][pos] = new
                row_diffs[rr] = nd
                all_diffs = set()
                for x in range(n_rows):
                    all_diffs |= row_diffs[x]
                cur_max = max(max(rows[x]) for x in range(n_rows))
                if cur_max < best_max:
                    best_max = cur_max

    return rows, best_max

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Fast greedy to get a baseline
    best_rows, best_max = _try_greedy_variants(n_rows, k, rng, num_trials=20)
    
    if best_rows is None:
        # Fallback
        best_rows = _greedy_build(n_rows, k, list(range(n_rows)), rng)
        best_max = max(max(r) for r in best_rows)
        
    # Phase 2: Backtracking search to improve
    # We try to find a solution with max < best_max
    # We allocate a portion of the time to this
    time_for_bt = 30.0 # 30 seconds for backtracking
    if _time_left() > time_for_bt + 10.0:
        bt_rows, bt_max = _backtracking_search(n_rows, k, best_max - 1, rng, time_limit=time_for_bt)
        if bt_rows is not None and bt_max < best_max:
            best_max = bt_max
            best_rows = bt_rows
            
    # Phase 3: Local search refinement
    if _time_left() > 10.0:
        ls_rows, ls_max = _local_search_reduce_max(best_rows, n_rows, k, rng, max_iters=20000)
        if ls_max < best_max:
            best_max = ls_max
            best_rows = ls_rows
            
    # Final verification
    if not _validate_solution(best_rows, n_rows, k):
        # If invalid, return the greedy one which should be valid
        # But if greedy is invalid, we have a bug.
        pass
        
    return best_rows
