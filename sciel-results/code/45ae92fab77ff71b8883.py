"""
Combines Row-Level ILS with Randomized Greedy Construction.
Strategy:
1. Generate initial solutions using deterministic and randomized greedy.
2. Use Iterated Local Search (ILS) where the "move" is replacing an entire row 
   with a new valid row found via randomized greedy, constrained by the 
   difference sets of all other rows.
3. Specifically target the reduction of the global maximum by attempting to 
   re-solve the row(s) contributing to the maximum with a tighter upper bound.
"""
import time
import random
import math
import sys

# EVOLVE-BLOCK-START

def _row_diffs_set(row):
    """Calculate the set of all pairwise differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s

def _is_sidon(row):
    """Check if a row is a valid Sidon sequence (all internal diffs distinct)."""
    n = len(row)
    seen = set()
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            d = row[j] - ri
            if d in seen:
                return False
            seen.add(d)
    return True

def _valid(rows, n_rows, k):
    """Check global validity: all rows valid, all diffs distinct globally."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        # Check internal Sidon property
        if not _is_sidon(row):
            return False
        # Check global distinctness
        diffs = _row_diffs_set(row)
        if diffs & used:
            return False
        used |= diffs
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _greedy_row_with_update(used, k, max_val=None, rng=None, bias_factor=1.0):
    """
    Build a single row greedily (optionally randomized).
    'used' is the set of forbidden differences from other rows.
    Returns (row, updated_used_set) or (None, used) if failed.
    bias_factor > 1.0 biases towards smaller values (more conservative).
    bias_factor < 1.0 biases towards larger values (more exploratory).
    """
    row = [0]
    x = 0
    limit = max_val if max_val is not None else 10000
    internal_used = set() # Diffs within this row so far
    
    while len(row) < k + 1:
        if rng is None:
            # Deterministic: pick smallest valid x
            x += 1
            if x > limit:
                return None, used
            new_diffs = [x - a for a in row]
            # Check internal consistency
            if len(set(new_diffs)) != len(new_diffs):
                continue
            # Check against global used
            if any(d in used for d in new_diffs):
                continue
            # Check against internal used (redundant if we check set size and add to used, 
            # but explicit check helps clarity)
            if any(d in internal_used for d in new_diffs):
                continue
            
            row.append(x)
            internal_used |= set(new_diffs)
            used |= set(new_diffs)
        else:
            # Randomized: find candidates in a window
            valid_xs = []
            scan = x + 1
            # Limit scan window for speed, but expand if needed
            window_size = 20
            end_scan = min(x + window_size, limit)
            
            while scan <= end_scan:
                new_diffs = [scan - a for a in row]
                if len(set(new_diffs)) == len(new_diffs):
                    if not any(d in used for d in new_diffs) and not any(d in internal_used for d in new_diffs):
                        valid_xs.append(scan)
                scan += 1
            
            if not valid_xs:
                # Fallback: scan further up to limit
                scan = x + 1
                while scan <= limit:
                    new_diffs = [scan - a for a in row]
                    if len(set(new_diffs)) == len(new_diffs):
                        if not any(d in used for d in new_diffs) and not any(d in internal_used for d in new_diffs):
                            valid_xs.append(scan)
                            break # Just take the first one if we had to scan far
                    scan += 1
                if not valid_xs:
                    return None, used
            
            # Weighted pick: bias towards smaller values
            # Weights decay exponentially with index
            weights = []
            for i in range(len(valid_xs)):
                weights.append(math.exp(-i / bias_factor))
            
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0.0
            chosen = valid_xs[-1]
            for v, w in zip(valid_xs, weights):
                cum += w
                if r <= cum:
                    chosen = v
                    break
            
            row.append(chosen)
            new_diffs = [chosen - a for a in row[:-1]]
            internal_used |= set(new_diffs)
            used |= set(new_diffs)
            x = chosen
            
    return row, used

def _build_greedy(n_rows, k, rng=None):
    """Fast greedy: each row takes smallest (or random-window) next mark with no diff collision."""
    used = set()
    rows = []
    for _ in range(n_rows):
        # Use a slightly randomized greedy for diversity if rng provided
        bias = 1.0 if rng is None else 1.5 # More biased to small for init
        row, used = _greedy_row_with_update(used, k, rng=rng, bias_factor=bias)
        if row is None:
            # Deterministic fallback
            row, used = _greedy_row_with_update(used, k, rng=None)
        if row is None:
            return None
        rows.append(row)
    return rows

def _local_optimize_row(rows, idx, n_rows, k, max_limit, rng):
    """
    Try to replace row[idx] with a new row that has max <= max_limit.
    Returns the new row if found and valid, else None.
    """
    # Collect forbidden diffs from all other rows
    forbidden = set()
    for i in range(n_rows):
        if i != idx:
            forbidden |= _row_diffs_set(rows[i])
    
    # Attempt to build a new row
    # We try a few times with different random seeds/biases
    for attempt in range(10):
        # Bias towards smaller values to fit within max_limit
        # If max_limit is tight, we need to be very greedy
        bias = 0.5 + (attempt * 0.1) # Start conservative, get slightly more exploratory
        new_row, _ = _greedy_row_with_update(forbidden, k, max_val=max_limit, rng=rng, bias_factor=bias)
        if new_row is not None:
            # Verify it's actually valid (greedy should ensure this, but double check)
            if _is_sidon(new_row):
                return new_row
    return None

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Phase 1: Initialization
    inits = []
    # Deterministic
    init_det = _build_greedy(n_rows, k, rng=None)
    if init_det:
        inits.append(init_det)
    
    # Randomized
    for s in range(10):
        rng = random.Random(12345 + s)
        init_r = _build_greedy(n_rows, k, rng=rng)
        if init_r:
            inits.append(init_r)
    
    if not inits:
        # Fallback: manual construction if greedy fails (unlikely for small k)
        # Simple greedy without randomization
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used and len(new_diffs) == len(row):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        inits.append(rows)

    # Sort by max entry
    inits.sort(key=_max_entry)
    best_rows = inits[0]
    best_val = _max_entry(best_rows)
    
    # Phase 2: ILS (Iterated Local Search)
    # We maintain a pool of good solutions or just iteratively improve the best.
    # Given time constraints, we focus on deep optimization of the best solution.
    
    rng = random.Random(42)
    current_rows = [r[:] for r in best_rows]
    current_val = best_val
    
    # Pre-compute diff sets for current rows for quick access
    current_diffs = [_row_diffs_set(r) for r in current_rows]
    
    no_improve_count = 0
    max_no_improve = 500 # Restart logic trigger
    
    while time.time() < deadline - 5.0:
        if no_improve_count > max_no_improve:
            # Perturb: Pick a random row and replace it with a fresh greedy row 
            # (possibly with a higher limit to allow escape from local minima)
            perturb_idx = rng.randrange(n_rows)
            # Use a high limit to ensure we find *a* valid row, even if it's worse
            # This resets that row to a "greedy best" state given others
            new_row, _ = _greedy_row_with_update(
                set().union(*[d for i, d in enumerate(current_diffs) if i != perturb_idx]), 
                k, 
                max_val=200, # Generous limit
                rng=rng
            )
            if new_row:
                current_rows[perturb_idx] = new_row
                current_diffs[perturb_idx] = _row_diffs_set(new_row)
                current_val = _max_entry(current_rows)
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve_count = 0
                else:
                    no_improve_count = 0 # Reset counter on perturbation
            else:
                # If perturbation failed, just reset counter to avoid infinite loop
                no_improve_count = 0
            continue
            
        # Local Optimization Step:
        # Try to reduce the global max.
        # Identify the row(s) with the maximum entry.
        max_val_current = current_val
        rows_at_max = [i for i in range(n_rows) if max(current_rows[i]) == max_val_current]
        
        # Try to improve one of the rows at the max
        if not rows_at_max:
            # Should not happen
            break
            
        target_idx = rng.choice(rows_at_max)
        
        # Attempt to find a row with max < max_val_current
        # We try limits: max_val_current - 1, max_val_current - 2, etc.
        # But we must ensure we don't spend too much time.
        # Let's try to find a row with max <= max_val_current - 1 first.
        
        # Collect forbidden diffs
        forbidden = set()
        for i in range(n_rows):
            if i != target_idx:
                forbidden |= current_diffs[i]
        
        improved = False
        # Try to find a better row
        # We try a few iterations with randomized greedy
        for attempt in range(20):
            # We want max < max_val_current
            # Let's try max_val_current - 1
            # If that fails, maybe max_val_current - 2? 
            # Actually, just trying max_val_current - 1 is a good heuristic.
            # If we can't find one, we might be stuck.
            
            # Randomized greedy with tight bound
            new_row, _ = _greedy_row_with_update(
                forbidden, 
                k, 
                max_val=max_val_current - 1, 
                rng=rng,
                bias_factor=1.0 # Balanced
            )
            
            if new_row is not None:
                # Verify
                if _is_sidon(new_row):
                    new_max = max(new_row)
                    # Since max_val was max_val_current - 1, new_max <= max_val_current - 1
                    # This strictly improves the global max
                    current_rows[target_idx] = new_row
                    current_diffs[target_idx] = _row_diffs_set(new_row)
                    current_val = _max_entry(current_rows)
                    
                    if current_val < best_val:
                        best_val = current_val
                        best_rows = [r[:] for r in current_rows]
                        no_improve_count = 0
                        improved = True
                    else:
                        # This shouldn't happen if logic is correct, but if current_val was
                        # determined by another row, and we fixed this row, global max might
                        # not change if another row is also at max.
                        # But we picked target_idx from rows_at_max.
                        # If there are multiple rows at max, fixing one might not lower global max.
                        # So current_val might stay same.
                        no_improve_count += 1
                    break
        
        if not improved:
            no_improve_count += 1
            
        # Occasionally, try to optimize a non-max row to see if it allows
        # a max row to be improved later? 
        # For now, stick to max row optimization as it directly impacts score.
        
    # Final Validation
    if not _valid(best_rows, n_rows, k):
        # Fallback to the best valid init
        for init in inits:
            if _valid(init, n_rows, k):
                return init
        # If all fail (should not happen), return the first one (which was valid in init phase)
        return inits[0]

    return best_rows

# EVOLVE-BLOCK-END
