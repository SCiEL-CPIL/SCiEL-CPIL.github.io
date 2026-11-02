"""Constraint-satisfaction search for the difference-triangle problem using Hybrid SA."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _greedy_fallback(n_rows, k):
    """Original greedy: always valid, used as safety net."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _scope(rows):
    return max(max(r) for r in rows)

def _compute_row_diffs(row):
    """Compute set of positive differences for a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _generate_row_randomized(forbidden, k, rng, max_val_hint):
    """
    Try to generate a valid row of size k+1 starting with 0.
    The row must not use any differences in 'forbidden'.
    Uses a randomized greedy/backtracking approach.
    Returns the row if successful, None otherwise.
    """
    row = [0]
    current_max = 0
    
    # We want to keep the row compact.
    # We'll try to pick the next smallest valid integer.
    # To introduce randomness, we might skip some valid candidates or try slightly larger ones.
    
    # A simple backtracking search for the row
    def backtrack(row, start_val):
        if len(row) == k + 1:
            return row[:]
        
        # Optimization: If we are getting too big, prune
        if start_val > max_val_hint * 1.5:
            return None

        # Try values starting from start_val
        # We can randomize the order slightly or just go sequential with some skips
        # Sequential is usually best for minimizing scope.
        
        # To add randomness, we can occasionally jump ahead
        # But for finding *any* valid row, sequential is robust.
        # We will use a "randomized sequential" where we pick the first valid one,
        # but we might restart the whole row generation if we get stuck in a bad spot.
        
        # Actually, for this specific move, we want the *best* (smallest) row possible given the forbidden set.
        # So greedy sequential is the right call for the "Regenerate" move.
        # The randomness comes from *which* row we regenerate and the *initial* state.
        
        for val in range(start_val, max_val_hint + 1):
            new_diffs = {val - a for a in row}
            if new_diffs & forbidden:
                continue
            
            # Also check if val is already in row (should not happen since we iterate increasing)
            # And check internal consistency (handled by new_diffs)
            
            row.append(val)
            result = backtrack(row, val + 1)
            if result is not None:
                return result
            row.pop()
            
        return None

    # We need a reasonable max_val_hint. 
    # The current scope is a good upper bound.
    # If we fail, we can retry with a larger hint.
    return backtrack(row, 1)

def construct(n_rows, k):
    deadline = time.time() + 115.0  # Safety margin before 120s kill
    rng = random.Random(42)
    
    # Phase 1: Greedy Initialization
    best_rows = _greedy_fallback(n_rows, k)
    best_scope = _scope(best_rows)
    
    # Try to improve initialization with a few quick random restarts of greedy
    for _ in range(10):
        # Shuffle the order of greedy construction? No, greedy is deterministic.
        # We can try a "randomized greedy" where we pick the next valid element with some probability of skipping.
        # But standard greedy is a good baseline.
        pass
        
    # State for Hybrid SA
    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope
    
    # Pre-compute global used differences
    def get_global_used(rows):
        used = set()
        for r in rows:
            used |= _compute_row_diffs(r)
        return used

    global_used = get_global_used(current_rows)
    
    # SA Parameters
    T = 100.0
    T_min = 1.0
    cooling_rate = 0.995
    
    start_time = time.time()
    
    while time.time() < deadline:
        # Move: Regenerate a random row
        r_idx = rng.randrange(n_rows)
        old_row = current_rows[r_idx]
        old_row_diffs = _compute_row_diffs(old_row)
        
        # Compute forbidden set: all diffs from other rows
        # We can compute this incrementally or just recompute (n_rows is small)
        forbidden = set()
        for i in range(n_rows):
            if i == r_idx:
                continue
            forbidden |= _compute_row_diffs(current_rows[i])
            
        # Try to generate a new row
        # Use current_scope as a hint. If we find a row with max < current_scope, great.
        # If we find a row with max <= current_scope, it's a candidate.
        # We want the *smallest* possible row to minimize scope.
        
        new_row = _generate_row_randomized(forbidden, k, rng, current_scope)
        
        if new_row is None:
            # If we couldn't find a row within current_scope, try with a larger limit?
            # Or just skip this move.
            # Sometimes the current row is the only valid one.
            T = max(T_min, T * cooling_rate)
            continue
            
        new_row_diffs = _compute_row_diffs(new_row)
        
        # The new row is valid by construction (diffs disjoint from forbidden).
        # We just need to check if it's valid against itself (it is, by construction)
        # and if it improves the scope.
        
        new_scope = max(new_row)
        
        # Calculate change in global scope
        # The new global scope is max(new_scope, max of other rows)
        other_max = 0
        for i in range(n_rows):
            if i == r_idx:
                continue
            other_max = max(other_max, max(current_rows[i]))
            
        new_global_scope = max(new_scope, other_max)
        old_global_scope = current_scope
        
        delta_E = new_global_scope - old_global_scope
        
        # Acceptance
        accept = False
        if delta_E <= 0:
            accept = True
        else:
            if T > 1e-9:
                prob = math.exp(-delta_E / T)
                accept = rng.random() < prob
            else:
                accept = False
                
        if accept:
            # Update state
            current_rows[r_idx] = new_row
            current_scope = new_global_scope
            
            # Update global_used
            # global_used = (global_used - old_row_diffs) | new_row_diffs
            # Since we know they are disjoint from forbidden (which is global_used - old_row_diffs),
            # we can just update.
            global_used -= old_row_diffs
            global_used |= new_row_diffs
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]
                
            T = max(T_min, T * cooling_rate)
        else:
            T = max(T_min, T * cooling_rate)
            
        # Safety check
        if time.time() > deadline - 1.0:
            break
            
    return best_rows

# EVOLVE-BLOCK-END
