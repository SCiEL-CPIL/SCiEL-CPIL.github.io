"""Difference Triangle Set (4,7) solver.

Strategy:
- Bitmask-based difference tracking for O(1) collision checks.
- Backtracking row construction with scope constraints.
- Fast local compaction using bitmasks.
- Simulated annealing with bitmask-accelerated moves.
"""
import time
import random
import sys

def _mask_for_row(row):
    """Compute the bitmask of all positive differences in a row."""
    mask = 0
    L = len(row)
    for i in range(L):
        for j in range(i + 1, L):
            d = row[j] - row[i]
            mask |= (1 << d)
    return mask

def _diffs_mask_for_pos(row, pos, L):
    """Compute bitmask of differences involving row[pos]."""
    mask = 0
    for i in range(L):
        if i < pos:
            mask |= (1 << (row[pos] - row[i]))
        elif i > pos:
            mask |= (1 << (row[i] - row[pos]))
    return mask

def _can_set_mask(row, pos, new_val, used_mask, L):
    """Check if setting row[pos] to new_val is valid using bitmask."""
    if pos > 0 and new_val <= row[pos - 1]:
        return False
    if pos < L - 1 and new_val >= row[pos + 1]:
        return False
    
    # Calculate new differences mask for this position
    new_diffs_mask = 0
    for i in range(L):
        if i < pos:
            new_diffs_mask |= (1 << (new_val - row[i]))
        elif i > pos:
            new_diffs_mask |= (1 << (row[i] - new_val))
    
    # Remove old differences for this position from used_mask to check against others
    old_diffs_mask = _diffs_mask_for_pos(row, pos, L)
    other_mask = used_mask & ~old_diffs_mask
    
    # If any new diff collides with an existing diff from other elements, it's invalid
    if new_diffs_mask & other_mask:
        return False
    return True

def _apply_set_mask(row, pos, new_val, used_mask, L):
    """Apply setting row[pos] to new_val, returning updated used_mask."""
    old_diffs_mask = _diffs_mask_for_pos(row, pos, L)
    new_diffs_mask = 0
    for i in range(L):
        if i < pos:
            new_diffs_mask |= (1 << (new_val - row[i]))
        elif i > pos:
            new_diffs_mask |= (1 << (row[i] - new_val))
    
    # Update mask: remove old, add new
    new_used_mask = (used_mask & ~old_diffs_mask) | new_diffs_mask
    row[pos] = new_val
    return new_used_mask

def _fast_compact_row_mask(row, used_mask, L):
    """Compact a single row in-place using bitmask checks. Returns (changed, new_used_mask)."""
    changed = False
    for pos in range(L - 1, 0, -1):
        old_val = row[pos]
        lower = row[pos - 1] + 1
        if lower >= old_val:
            continue
        
        old_diffs_mask = _diffs_mask_for_pos(row, pos, L)
        other_mask = used_mask & ~old_diffs_mask
        
        for new_val in range(lower, old_val):
            if pos < L - 1 and new_val >= row[pos + 1]:
                break
            
            new_diffs_mask = 0
            for i in range(L):
                if i < pos:
                    new_diffs_mask |= (1 << (new_val - row[i]))
                elif i > pos:
                    new_diffs_mask |= (1 << (row[i] - new_val))
            
            if not (new_diffs_mask & other_mask):
                # Valid move
                used_mask = (used_mask & ~old_diffs_mask) | new_diffs_mask
                row[pos] = new_val
                changed = True
                break
    return changed, used_mask

def _local_compact_mask(rows, n_rows, k, time_limit):
    """Iteratively compact each element using bitmask checks."""
    work = [list(r) for r in rows]
    L = k + 1
    
    # Build global used mask
    used_mask = 0
    for row in work:
        used_mask |= _mask_for_row(row)
        
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for r_idx in range(n_rows):
            row = work[r_idx]
            c, used_mask = _fast_compact_row_mask(row, used_mask, L)
            if c:
                improved = True
    return work, used_mask

def _backtrack_row(row_idx, used_mask, max_scope, k, rng, time_limit):
    """
    Backtrack to construct a single row that fits in max_scope and uses no diffs in used_mask.
    Returns (row, new_used_mask) or (None, used_mask) if failed.
    """
    L = k + 1
    t_start = time.time()
    
    def build(pos, last_val, current_row, current_mask):
        if time.time() - t_start > time_limit:
            return None
        if pos == L:
            return list(current_row)
        
        # Prune: we need L-pos more elements, each at least 1 apart
        # So the last element must be at most max_scope - (L - pos - 1)
        max_x = max_scope - (L - pos - 1)
        if max_x <= last_val:
            return None
            
        # Candidates: try smaller values first to minimize scope, but shuffle slightly for diversity
        # We iterate from last_val+1 to max_x
        # Optimization: only check candidates that don't immediately collide
        candidates = []
        for x in range(last_val + 1, max_x + 1):
            # Check diffs with current_row
            valid = True
            new_diffs_mask = 0
            for a in current_row:
                d = x - a
                bit = 1 << d
                if current_mask & bit or used_mask & bit:
                    valid = False
                    break
                new_diffs_mask |= bit
            if valid:
                candidates.append((x, new_diffs_mask))
                
        if not candidates:
            return None
            
        # Shuffle a window of candidates to allow some exploration
        # Prefer smaller x, but mix in some larger ones
        n_take = min(10, len(candidates))
        window = candidates[:n_take]
        if len(candidates) > n_take and rng.random() < 0.5:
            # Sample a few from the rest
            extra_idx = rng.sample(range(n_take, len(candidates)), min(5, len(candidates) - n_take))
            window.extend(candidates[i] for i in extra_idx)
        rng.shuffle(window)
        
        for x, new_diffs_mask in window:
            new_mask = current_mask | new_diffs_mask
            res = build(pos + 1, x, current_row + [x], new_mask)
            if res is not None:
                return res
        return None

    row = build(1, 0, [0], 0)
    if row is None:
        return None, used_mask
        
    final_mask = _mask_for_row(row)
    # Ensure no collision with global used_mask (should be guaranteed by build, but double check)
    if final_mask & used_mask:
        return None, used_mask
        
    return row, used_mask | final_mask

def _rebuild_row_mask(r_idx, work, used_mask, L, rng, max_scope, time_limit):
    """Rebuild a row using backtracking."""
    row = work[r_idx]
    old_mask = _mask_for_row(row)
    # Remove old row's diffs from global mask
    global_mask_without_row = used_mask & ~old_mask
    
    # Try to build a new row
    new_row, new_global_mask = _backtrack_row(r_idx, global_mask_without_row, max_scope, L-1, rng, time_limit)
    
    if new_row is not None:
        work[r_idx] = new_row
        return True, new_global_mask
    else:
        # Restore
        return False, used_mask

def _simulated_annealing_mask(rows, n_rows, k, rng, time_limit):
    """Simulated annealing with bitmask acceleration."""
    work = [list(r) for r in rows]
    L = k + 1
    used_mask = 0
    for row in work:
        used_mask |= _mask_for_row(row)
        
    current_max = max(max(r) for r in work)
    best_rows = [list(r) for r in work]
    best_max = current_max
    
    start_temp = max(1, current_max // 8)
    end_temp = 0.05
    cooling = 0.9995
    t = start_temp
    
    while time.time() < time_limit and t > end_temp:
        move_type = rng.random()
        
        if move_type < 0.40:
            # Move 1: Decrease a random element (Compaction move)
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            lower = row[pos - 1] + 1
            if lower >= old_val:
                t *= cooling
                continue
                
            # Try to set to a random value in [lower, old_val-1]
            # Or just try the smallest feasible one? Random helps escape local minima.
            new_val = rng.randint(lower, old_val - 1)
            if new_val >= old_val:
                t *= cooling
                continue
                
            if _can_set_mask(row, pos, new_val, used_mask, L):
                used_mask = _apply_set_mask(row, pos, new_val, used_mask, L)
                # Try to compact the rest of the row
                c, used_mask = _fast_compact_row_mask(row, used_mask, L)
                new_max = max(max(r) for r in work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]

        elif move_type < 0.65:
            # Move 2: Increase a random element (Perturbation)
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            upper_limit = old_val + rng.randint(1, max(1, current_max // 4))
            new_val = rng.randint(old_val + 1, upper_limit)
            if new_val <= old_val:
                t *= cooling
                continue
                
            if _can_set_mask(row, pos, new_val, used_mask, L):
                used_mask = _apply_set_mask(row, pos, new_val, used_mask, L)
                c, used_mask = _fast_compact_row_mask(row, used_mask, L)
                new_max = max(max(r) for r in work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]

        elif move_type < 0.85:
            # Move 3: Rebuild a row with backtracking
            r_idx = rng.randint(0, n_rows - 1)
            # Try to fit in current best max
            success, used_mask = _rebuild_row_mask(r_idx, work, used_mask, L, rng, best_max, time.time() + 0.5)
            if success:
                new_max = max(max(r) for r in work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]
                    current_max = best_max

        else:
            # Move 4: Full recompact
            for r_idx in range(n_rows):
                c, used_mask = _fast_compact_row_mask(work[r_idx], used_mask, L)
            new_max = max(max(r) for r in work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
                current_max = best_max

        t *= cooling

    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    time_limit = start_time + 115  # Safety margin
    
    L = k + 1
    rng = random.Random(42)
    
    # 1. Initial construction using Backtracking with a loose bound
    # Counting bound: n_rows * k * (k+1) / 2
    # For 4,7: 4 * 28 = 112. Best known is 134.
    # Let's start with a bound of 140 and try to tighten.
    init_bound = 140
    rows = []
    used_mask = 0
    
    success = False
    for _ in range(100): # Try multiple seeds/bounds
        # Try to build all rows
        trial_rows = []
        trial_mask = 0
        valid = True
        for i in range(n_rows):
            row, new_mask = _backtrack_row(i, trial_mask, init_bound, k, rng, time.time() + 2.0)
            if row is None:
                valid = False
                break
            trial_rows.append(row)
            trial_mask = new_mask
        
        if valid:
            rows = trial_rows
            used_mask = trial_mask
            success = True
            break
            
    if not success:
        # Fallback to simple greedy if backtracking fails (shouldn't happen)
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < L:
                x += 1
                new_diffs_mask = 0
                valid = True
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if used_mask & bit:
                        valid = False
                        break
                    new_diffs_mask |= bit
                if valid:
                    row.append(x)
                    used_mask |= new_diffs_mask
            rows.append(row)
            
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in best_rows)
    
    # 2. Local Compaction
    if time.time() < time_limit - 30:
        compacted, _ = _local_compact_mask(best_rows, n_rows, k, time_limit - 30)
        cmax = max(max(r) for r in compacted)
        if cmax < best_max:
            best_rows = compacted
            best_max = cmax
            
    # 3. Simulated Annealing
    if time.time() < time_limit - 5:
        sa_result = _simulated_annealing_mask(best_rows, n_rows, k, rng, time_limit - 5)
        smax = max(max(r) for r in sa_result)
        if smax < best_max:
            best_rows = sa_result
            best_max = smax
            
    # 4. Final Validation (Slow check just in case)
    # We trust the bitmask logic, but a final set-check is cheap for 4x8
    used = set()
    valid = True
    for row in best_rows:
        for i in range(L):
            for j in range(i + 1, L):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                    break
                used.add(d)
        if not valid:
            break
            
    if not valid:
        # Emergency fallback
        return _greedy_construct_fallback(n_rows, k)
        
    return best_rows

def _greedy_construct_fallback(n_rows, k):
    """Simple greedy fallback if everything fails."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

# Keep the original signature requirement
# The task asks for construct(n_rows, k)
