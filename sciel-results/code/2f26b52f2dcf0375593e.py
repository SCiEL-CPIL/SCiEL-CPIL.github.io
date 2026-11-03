"""Difference Triangle Set (4,7) solver.

Strategy:
- Fast greedy baseline for an initial valid configuration.
- Difference-space backtracking: search over which differences each row uses,
  with constraint propagation and interval-based pruning.
- Local compaction and simulated annealing for final polish.
"""
import time
import random

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
    
    new_diffs_mask = 0
    for i in range(L):
        if i < pos:
            new_diffs_mask |= (1 << (new_val - row[i]))
        elif i > pos:
            new_diffs_mask |= (1 << (row[i] - new_val))
    
    old_diffs_mask = _diffs_mask_for_pos(row, pos, L)
    other_mask = used_mask & ~old_diffs_mask
    
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
                used_mask = (used_mask & ~old_diffs_mask) | new_diffs_mask
                row[pos] = new_val
                changed = True
                break
    return changed, used_mask

def _local_compact_mask(rows, n_rows, k, time_limit):
    """Iteratively compact each element using bitmask checks."""
    work = [list(r) for r in rows]
    L = k + 1
    
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

def _backtrack_row(used_mask, max_scope, k, rng, time_limit):
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
        
        max_x = max_scope - (L - pos - 1)
        if max_x <= last_val:
            return None
            
        candidates = []
        for x in range(last_val + 1, max_x + 1):
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
            
        n_take = min(10, len(candidates))
        window = candidates[:n_take]
        if len(candidates) > n_take and rng.random() < 0.5:
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
    if final_mask & used_mask:
        return None, used_mask
        
    return row, used_mask | final_mask

def _rebuild_row_mask(r_idx, work, used_mask, L, rng, max_scope, time_limit):
    """Rebuild a row using backtracking."""
    row = work[r_idx]
    old_mask = _mask_for_row(row)
    global_mask_without_row = used_mask & ~old_mask
    
    new_row, new_global_mask = _backtrack_row(global_mask_without_row, max_scope, L-1, rng, time_limit)
    
    if new_row is not None:
        work[r_idx] = new_row
        return True, new_global_mask
    else:
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
                
            new_val = rng.randint(lower, old_val - 1)
            if new_val >= old_val:
                t *= cooling
                continue
                
            if _can_set_mask(row, pos, new_val, used_mask, L):
                used_mask = _apply_set_mask(row, pos, new_val, used_mask, L)
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

def _difference_space_backtrack(n_rows, k, max_scope, rng, time_limit):
    """
    Structural change: search over which differences each row uses,
    with constraint propagation to prune early.
    
    For each row, we need to choose C(k+1, 2) = 36 differences that:
    1. Are all distinct across rows
    2. Can be realized by an increasing sequence of k+1 integers starting at 0
    3. The largest integer is <= max_scope
    
    We use a different representation: instead of constructing sequences,
    we directly assign differences to rows with feasibility checks.
    """
    L = k + 1  # 9
    num_diffs_per_row = L * (L - 1) // 2  # 36
    total_diffs = n_rows * num_diffs_per_row  # 144
    
    t_start = time.time()
    
    # For a given set of differences, check if they can form a valid row
    # A row [0, a1, a2, ..., a8] has differences:
    # a1, a2, ..., a8, a2-a1, a3-a1, ..., a8-a1, a3-a2, ..., a8-a2, ..., a8-a7
    # Total 36 differences
    
    # Key insight: if we know all 36 differences, we can reconstruct the row
    # The smallest difference must be a1 (the first gap)
    # Then a2 = a1 + (a2-a1), etc.
    # But we need to verify consistency
    
    def _can_form_row(diffs, max_scope):
        """Check if a set of 36 differences can form a valid row."""
        if len(diffs) != 36:
            return None
        
        # Sort differences
        sorted_diffs = sorted(diffs)
        
        # The row must start with 0, so the differences include all ai and all aj-ai
        # The smallest difference must be one of the consecutive gaps
        # Let's try to reconstruct: the row is [0, x1, x2, ..., x8]
        # Differences are: x1, x2, ..., x8, x2-x1, x3-x1, ..., x8-x1, x3-x2, ..., x8-x2, ..., x8-x7
        
        # Strategy: x1 must be the smallest difference (since all other differences >= x1)
        # Actually, x2-x1 could be smaller than x1? No, x2 > x1 > 0, so x2-x1 >= 1
        # But x1 could be larger than x2-x1. Example: [0, 5, 7] -> diffs: 5, 7, 2
        # So the smallest diff is 2 = x2-x1, not x1=5
        
        # Better approach: try all possible values for x1
        # x1 must be in diffs, and x1 >= 1
        # Then x2 = x1 + d for some d in diffs where d = x2-x1
        # This is complex; let's use a different strategy
        
        # Alternative: use the fact that the sum of certain differences gives constraints
        # For now, use a simpler check: verify that diffs can be partitioned into
        # 8 "row values" and 28 "gap differences" consistently
        
        # Practical approach: try to reconstruct by assuming x1 is the smallest diff
        # that, when used as x1, allows consistent reconstruction
        
        best_row = None
        best_max = float('inf')
        
        # Try each possible x1 (the first element after 0)
        # x1 must be in diffs
        for x1_candidate in sorted_diffs:
            if x1_candidate > max_scope:
                break
            
            # Assume x1 = x1_candidate
            # Then x2-x1 must be in diffs, and x2 = x1 + (x2-x1)
            # Let's try to build the row greedily
            
            row = [0, x1_candidate]
            used_in_row = {x1_candidate}
            remaining = set(diffs) - used_in_row
            
            # Now we need to find x2, x3, ..., x8
            # x2 must be such that x2-x1 is in remaining, and x2 is in remaining
            # Actually, x2 is in diffs (as the difference from 0)
            # And x2-x1 is also in diffs
            
            # Let's try all possible x2 values
            for x2_candidate in remaining:
                if x2_candidate <= x1_candidate:
                    continue
                if x2_candidate > max_scope:
                    break
                gap12 = x2_candidate - x1_candidate
                if gap12 not in remaining:
                    continue
                # Check that gap12 != x1_candidate (already used) and gap12 != x2_candidate
                if gap12 in used_in_row or gap12 == x2_candidate:
                    continue
                
                row2 = [0, x1_candidate, x2_candidate]
                used2 = {x1_candidate, x2_candidate, gap12}
                remaining2 = set(diffs) - used2
                
                # Continue building...
                # This is getting complex; let's use a different approach
                
                # Actually, let's use a more direct reconstruction:
                # Given all 36 differences, the row values are:
                # 0, x1, x2, ..., x8 where xi = sum of some subset of "atomic" gaps
                # The atomic gaps are g1 = x1, g2 = x2-x1, ..., g8 = x8-x7
                # Then xi = g1 + g2 + ... + gi
                # And the differences are all sums of consecutive atomic gaps
                
                # So we need to find 8 atomic gaps g1,...,g8 such that:
                # The set {sum(gi..gj) for 1<=i<=j<=8} equals the given 36 diffs
                # And sum(gi) = x8 <= max_scope
                
                # This is a different problem! Let's try to solve it
                # The 36 differences are all contiguous sums of the 8 atomic gaps
                # We can identify the atomic gaps as the "minimal" differences
                # in a certain sense
                
                # For now, let's use a simpler heuristic: 
                # The 8 row values x1,...,x8 must all be in diffs
                # And x1 < x2 < ... < x8
                # And for each i<j, xj-xi must be in diffs
                # And the 28 differences xj-xi (i<j, not including xj itself) must be in diffs
                # Wait, xj is also a difference (from 0), so we have:
                # 8 differences: x1, x2, ..., x8 (from 0)
                # 28 differences: xj-xi for 1<=i<j<=8
                # Total 36
                
                # So: choose 8 values x1<...<x8 from diffs such that
                # all xj-xi for i<j are also in diffs
                # And x8 <= max_scope
                
                # Let's try this approach
                # Sort diffs
                # Try to find 8 values that work
                
                pass  # This is getting too complex for inline; let's use a different strategy
        
        # Let's use a simpler but effective approach:
        # Greedily reconstruct the row from the differences
        # The key insight: the differences include all "prefix sums" of atomic gaps
        # Let's find the atomic gaps by looking at the structure
        
        # Actually, let's just try all combinations of 8 values from the 36 diffs
        # That's C(36,8) = 30 million, too many
        
        # Better: use the fact that x1 must be in diffs, x2 must be in diffs, etc.
        # And xj - xi must be in diffs for all i<j
        
        # Let's use backtracking with pruning
        def _find_row_values(diffs_set, max_scope, n_vals=8):
            """Find n_vals increasing values from diffs_set such that all pairwise differences are in diffs_set."""
            sorted_vals = sorted(diffs_set)
            
            def backtrack(start_idx, chosen, last_val):
                if len(chosen) == n_vals:
                    return list(chosen)
                if time.time() - t_start > time_limit:
                    return None
                
                remaining_needed = n_vals - len(chosen)
                # Pruning: we need remaining_needed more values, each at least last_val + 1
                # And the largest must be <= max_scope
                if last_val + remaining_needed > max_scope:
                    return None
                
                for i in range(start_idx, len(sorted_vals)):
                    val = sorted_vals[i]
                    if val <= last_val:
                        continue
                    if val > max_scope:
                        break
                    
                    # Check that val - c is in diffs_set for all c in chosen
                    valid = True
                    for c in chosen:
                        if (val - c) not in diffs_set:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Pruning: check if we can still fit remaining_needed values
                    # The next value must be > val, and we need remaining_needed-1 more
                    # Each must differ from all chosen values by an amount in diffs_set
                    # Simple bound: val + (remaining_needed-1) <= max_scope
                    if val + (remaining_needed - 1) > max_scope:
                        continue
                    
                    res = backtrack(i + 1, chosen + [val], val)
                    if res is not None:
                        return res
                return None
            
            return backtrack(0, [], 0)
        
        row_values = _find_row_values(set(diffs), max_scope)
        if row_values is not None:
            row = [0] + row_values
            # Verify
            for i in range(9):
                for j in range(i+1, 9):
                    if row[j] - row[i] not in diffs_set:
                        return None
            if max(row) <= max_scope:
                return row
        
        return None
    
    # Now, the main search: assign differences to rows
    # We'll use a greedy + backtracking hybrid
    
    best_rows = None
    best_max = float('inf')
    
    # First, get an initial solution using the existing backtracker
    initial_rows = []
    used_mask = 0
    for _ in range(n_rows):
        row, new_mask = _backtrack_row(used_mask, max_scope, k, rng, time_limit - 5)
        if row is None:
            break
        initial_rows.append(row)
        used_mask = new_mask
    
    if len(initial_rows) == n_rows:
        best_rows = initial_rows
        best_max = max(max(r) for r in best_rows)
    
    # Now try to improve by rebuilding rows with the difference-space approach
    # For each row, try to find a better assignment of differences
    
    t_start_improve = time.time()
    improve_time_limit = time_limit - 10
    
    while time.time() < improve_time_limit and best_rows is not None:
        # Pick a random row to rebuild
        r_idx = rng.randint(0, n_rows - 1)
        
        # Compute the mask of differences used by other rows
        other_mask = 0
        for i in range(n_rows):
            if i != r_idx:
                other_mask |= _mask_for_row(best_rows[i])
        
        # Find available differences (not in other_mask, up to max_scope)
        available = []
        for d in range(1, max_scope + 1):
            if not (other_mask & (1 << d)):
                available.append(d)
        
        if len(available) < num_diffs_per_row:
            continue
        
        # Try to find a valid assignment of 36 differences to this row
        # Use random sampling + verification
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts and time.time() < improve_time_limit:
            attempts += 1
            
            # Sample 36 differences from available
            # Bias towards smaller differences to reduce scope
            sampled = rng.sample(available, min(36, len(available)))
            
            # Try to form a row from these differences
            row = _can_form_row(sampled, max_scope)
            if row is not None:
                # Check that this row doesn't conflict with others
                row_mask = _mask_for_row(row)
                if not (row_mask & other_mask):
                    # Update best_rows
                    new_rows = list(best_rows)
                    new_rows[r_idx] = row
                    new_max = max(max(r) for r in new_rows)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = new_rows
                        break
    
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    time_limit = start_time + 115  # Safety margin
    
    L = k + 1
    rng = random.Random(42)
    
    # 1. Initial construction using Backtracking with a loose bound
    init_bound = 140
    rows = []
    used_mask = 0
    
    success = False
    for _ in range(100):
        trial_rows = []
        trial_mask = 0
        valid = True
        for i in range(n_rows):
            row, new_mask = _backtrack_row(trial_mask, init_bound, k, rng, time.time() + 2.0)
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
    
    # 4. Final Validation
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
