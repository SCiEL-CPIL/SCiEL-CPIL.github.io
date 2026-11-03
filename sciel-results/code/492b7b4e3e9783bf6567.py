"""Difference Triangle Set (4,7) solver.

Combined strategy:
- Fast greedy baseline + many randomized greedy variants (with row-order shuffling, bias).
- Bitset-based difference tracking for O(1) conflict checks.
- Efficient local compaction (right-to-left, bitset-based difference checks).
- Targeted backtracking search with explicit scope reduction.
- Time-bounded simulated annealing with:
    * single-element decrease (compaction) moves
    * random element perturbations + recompact
    * full-row rebuilds (multiple strategies, bounded by current max)
    * full recompact passes
    * pair-swap moves between rows
- Safety margin before the 150 s wall-clock limit.
"""
import time
import random


def _check_valid(rows, k):
    """Return True iff all positive differences within rows are distinct."""
    used = set()
    for row in rows:
        L = len(row)
        for i in range(L):
            for j in range(i + 1, L):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _max_entry(rows):
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, rng=None):
    """Greedy ruler construction with optional randomization."""
    if rng is None:
        rng = random.Random(0)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if rng.random() < 0.02:
                x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _greedy_variant(n_rows, k, rng, skip_p=0.05, jump_p=0.03, shuffle_rows=False,
                    bias_p=0.0, bias_range=5):
    """Greedy with configurable skip, jump, shuffle, and bias parameters."""
    used = set()
    rows = []
    order = list(range(n_rows))
    if shuffle_rows:
        rng.shuffle(order)
    row_results = {}
    for r_idx in order:
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if rng.random() < jump_p:
                x += rng.randint(1, 4)
            if rng.random() < skip_p:
                continue
            if bias_p > 0 and rng.random() < bias_p:
                x += rng.randint(1, bias_range)
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        if len(row) != k + 1:
            return None
        row_results[r_idx] = row
    for r_idx in range(n_rows):
        rows.append(row_results[r_idx])
    return rows


def _build_used_set(rows, k):
    L = k + 1
    used = set()
    for row in rows:
        for i in range(L):
            for j in range(i + 1, L):
                used.add(row[j] - row[i])
    return used


def _get_diffs_for_pos(row, pos, L):
    diffs = set()
    for i in range(L):
        if i < pos:
            diffs.add(row[pos] - row[i])
        elif i > pos:
            diffs.add(row[i] - row[pos])
    return diffs


def _can_set(row, pos, new_val, used_set, L):
    if pos > 0 and new_val <= row[pos - 1]:
        return False
    if pos < L - 1 and new_val >= row[pos + 1]:
        return False
    old_diffs = _get_diffs_for_pos(row, pos, L)
    new_diffs = set()
    for i in range(L):
        if i < pos:
            new_diffs.add(new_val - row[i])
        elif i > pos:
            new_diffs.add(row[i] - new_val)
    other = used_set - old_diffs
    if new_diffs & other:
        return False
    return True


def _apply_set(row, pos, new_val, used_set, L):
    old_diffs = _get_diffs_for_pos(row, pos, L)
    new_diffs = set()
    for i in range(L):
        if i < pos:
            new_diffs.add(new_val - row[i])
        elif i > pos:
            new_diffs.add(row[i] - new_val)
    for d in old_diffs:
        used_set.discard(d)
    for d in new_diffs:
        used_set.add(d)
    row[pos] = new_val


def _fast_compact_row(row, used_set, L):
    """Compact a single row in-place, return True if any change made."""
    changed = False
    for pos in range(L - 1, 0, -1):
        old_val = row[pos]
        lower = row[pos - 1] + 1
        if lower >= old_val:
            continue
        old_diffs = _get_diffs_for_pos(row, pos, L)
        for new_val in range(lower, old_val):
            if pos < L - 1 and new_val >= row[pos + 1]:
                break
            new_diffs = set()
            for i in range(L):
                if i < pos:
                    new_diffs.add(new_val - row[i])
                elif i > pos:
                    new_diffs.add(row[i] - new_val)
            other = used_set - old_diffs
            if new_diffs & other:
                continue
            for d in old_diffs:
                used_set.discard(d)
            for d in new_diffs:
                used_set.add(d)
            row[pos] = new_val
            changed = True
            break
    return changed


def _local_compact(rows, n_rows, k, time_limit):
    """Iteratively compact each element to its smallest feasible value."""
    work = [list(r) for r in rows]
    used_set = _build_used_set(work, k)
    L = k + 1
    improved = True

    while improved and time.time() < time_limit:
        improved = False
        for r_idx in range(n_rows):
            row = work[r_idx]
            for pos in range(L - 1, 0, -1):
                old_val = row[pos]
                lower = row[pos - 1] + 1
                if lower >= old_val:
                    continue
                old_diffs = _get_diffs_for_pos(row, pos, L)
                for new_val in range(lower, old_val):
                    if pos < L - 1 and new_val >= row[pos + 1]:
                        break
                    new_diffs = set()
                    for i in range(L):
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        elif i > pos:
                            new_diffs.add(row[i] - new_val)
                    other = used_set - old_diffs
                    if new_diffs & other:
                        continue
                    for d in old_diffs:
                        used_set.discard(d)
                    for d in new_diffs:
                        used_set.add(d)
                    row[pos] = new_val
                    improved = True
                    break
    return work


def _rebuild_row(r_idx, work, used_set, L, rng, max_val):
    """Rebuild a row from scratch with multiple strategies, keeping max below max_val."""
    row = work[r_idx]
    old_diffs = set()
    for i in range(L):
        for j in range(i + 1, L):
            old_diffs.add(row[j] - row[i])
    for d in old_diffs:
        used_set.discard(d)

    best_new_row = None
    best_new_max = max_val + 1

    for attempt in range(60):
        strategy = rng.random()
        new_row = [0]
        x = 0
        valid = True

        if strategy < 0.35:
            # Standard greedy with small random skips
            while len(new_row) < L:
                x += 1
                if rng.random() < 0.08:
                    x += rng.randint(1, 3)
                new_diffs = {x - a for a in new_row}
                if new_diffs & used_set:
                    continue
                new_row.append(x)
                if x > best_new_max:
                    valid = False
                    break
        elif strategy < 0.60:
            # Aggressive jumps
            while len(new_row) < L:
                x += rng.randint(1, 5)
                new_diffs = {x - a for a in new_row}
                if new_diffs & used_set:
                    continue
                new_row.append(x)
                if x > best_new_max:
                    valid = False
                    break
        elif strategy < 0.80:
            # Biased: prefer larger values early to spread out
            while len(new_row) < L:
                x += rng.randint(1, 3)
                if rng.random() < 0.15:
                    x += rng.randint(2, 5)
                new_diffs = {x - a for a in new_row}
                if new_diffs & used_set:
                    continue
                new_row.append(x)
                if x > best_new_max:
                    valid = False
                    break
        else:
            # Conservative: step by 1, occasional skip
            while len(new_row) < L:
                x += 1
                if rng.random() < 0.04:
                    x += 1
                new_diffs = {x - a for a in new_row}
                if new_diffs & used_set:
                    continue
                new_row.append(x)
                if x > best_new_max:
                    valid = False
                    break

        if valid and len(new_row) == L:
            new_max = new_row[-1]
            if new_max < best_new_max:
                best_new_max = new_max
                best_new_row = list(new_row)

    if best_new_row is not None:
        for i in range(L):
            for j in range(i + 1, L):
                used_set.add(best_new_row[j] - best_new_row[i])
        work[r_idx] = best_new_row
        return True
    else:
        for d in old_diffs:
            used_set.add(d)
        work[r_idx] = row
        return False


def _try_pair_swap(work, used_set, L, r1, r2, pos1, pos2):
    """Try swapping elements at (r1,pos1) and (r2,pos2). Returns True if swap applied."""
    row1 = work[r1]
    row2 = work[r2]
    val1 = row1[pos1]
    val2 = row2[pos2]

    # Check ordering constraints after swap
    if r1 == r2:
        return False

    # After swap: row1[pos1] = val2, row2[pos2] = val1
    # Check row1 ordering
    if pos1 > 0 and val2 <= row1[pos1 - 1]:
        return False
    if pos1 < L - 1 and val2 >= row1[pos1 + 1]:
        return False
    # Check row2 ordering
    if pos2 > 0 and val1 <= row2[pos2 - 1]:
        return False
    if pos2 < L - 1 and val1 >= row2[pos2 + 1]:
        return False

    # Compute old diffs for both positions
    old_diffs1 = _get_diffs_for_pos(row1, pos1, L)
    old_diffs2 = _get_diffs_for_pos(row2, pos2, L)

    # Compute new diffs
    new_diffs1 = set()
    for i in range(L):
        if i < pos1:
            new_diffs1.add(val2 - row1[i])
        elif i > pos1:
            new_diffs1.add(row1[i] - val2)

    new_diffs2 = set()
    for i in range(L):
        if i < pos2:
            new_diffs2.add(val1 - row2[i])
        elif i > pos2:
            new_diffs2.add(row2[i] - val1)

    # Check for conflicts with other used diffs
    other = used_set - old_diffs1 - old_diffs2
    if (new_diffs1 | new_diffs2) & other:
        return False

    # Check for conflicts between the two new diff sets
    if new_diffs1 & new_diffs2:
        return False

    # Apply the swap
    for d in old_diffs1:
        used_set.discard(d)
    for d in old_diffs2:
        used_set.discard(d)
    for d in new_diffs1:
        used_set.add(d)
    for d in new_diffs2:
        used_set.add(d)

    row1[pos1] = val2
    row2[pos2] = val1
    return True


def _simulated_annealing(rows, n_rows, k, rng, time_limit):
    """Simulated annealing to improve the solution with pair-swap moves."""
    work = [list(r) for r in rows]
    L = k + 1
    used_set = _build_used_set(work, k)
    current_max = _max_entry(work)
    best_rows = [list(r) for r in work]
    best_max = current_max

    start_temp = max(1, current_max // 10)
    end_temp = 0.1
    cooling = 0.9997

    t = start_temp
    iter_count = 0

    while time.time() < time_limit and t > end_temp:
        iter_count += 1
        move_type = rng.random()

        if move_type < 0.30:
            # Single-element decrease (compaction)
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
            if _can_set(row, pos, new_val, used_set, L):
                _apply_set(row, pos, new_val, used_set, L)
                _fast_compact_row(row, used_set, L)
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]

        elif move_type < 0.50:
            # Random element perturbation + recompact
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            upper_limit = old_val + rng.randint(1, max(1, current_max // 4))
            new_val = rng.randint(old_val + 1, upper_limit)
            if new_val <= old_val:
                t *= cooling
                continue
            if _can_set(row, pos, new_val, used_set, L):
                _apply_set(row, pos, new_val, used_set, L)
                _fast_compact_row(row, used_set, L)
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]

        elif move_type < 0.65:
            # Full-row rebuild
            r_idx = rng.randint(0, n_rows - 1)
            _rebuild_row(r_idx, work, used_set, L, rng, current_max)
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
                current_max = best_max

        elif move_type < 0.80:
            # Pair-swap move between two rows
            if n_rows >= 2:
                r1, r2 = rng.sample(range(n_rows), 2)
                pos1 = rng.randint(1, L - 1)
                pos2 = rng.randint(1, L - 1)
                if _try_pair_swap(work, used_set, L, r1, r2, pos1, pos2):
                    # Compact both affected rows
                    _fast_compact_row(work[r1], used_set, L)
                    _fast_compact_row(work[r2], used_set, L)
                    new_max = _max_entry(work)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(r) for r in work]
                        current_max = best_max

        else:
            # Full recompact pass
            for r_idx in range(n_rows):
                _fast_compact_row(work[r_idx], used_set, L)
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
                current_max = best_max

        t *= cooling

    return best_rows


def _bitset_backtrack(rows, n_rows, k, rng, time_limit, target_scope=None):
    """Targeted backtracking search with bitset acceleration.
    
    Tries to reduce the maximum entry below target_scope by backtracking
    through rows in reverse order, using bitsets for fast conflict detection.
    """
    L = k + 1
    if target_scope is None:
        target_scope = _max_entry(rows) - 1
    
    work = [list(r) for r in rows]
    best_rows = [list(r) for r in work]
    best_max = _max_entry(work)
    
    # Bitset representation of used differences
    # We need bits up to target_scope
    used_bits = 0
    
    def get_diff_bitset(row):
        """Compute bitset of all positive differences in a row."""
        bits = 0
        for i in range(L):
            for j in range(i + 1, L):
                d = row[j] - row[i]
                if d <= target_scope:
                    bits |= (1 << d)
        return bits
    
    def set_diffs(row):
        """Return set of positive differences in a row."""
        diffs = set()
        for i in range(L):
            for j in range(i + 1, L):
                diffs.add(row[j] - row[i])
        return diffs
    
    def can_place(row, pos, val, other_diffs):
        """Check if val can be placed at pos in row given other_diffs (bitset)."""
        if pos > 0 and val <= row[pos - 1]:
            return False
        if pos < L - 1 and val >= row[pos + 1]:
            return False
        
        # Compute new differences involving position pos
        new_diffs = set()
        for i in range(L):
            if i < pos:
                d = val - row[i]
            elif i > pos:
                d = row[i] - val
            else:
                continue
            new_diffs.add(d)
        
        # Check conflicts
        for d in new_diffs:
            if d <= target_scope and (other_diffs >> d) & 1:
                return False
        
        return True
    
    def backtrack(r_idx, pos, current_used_bits, depth=0):
        """Backtrack to place elements in row r_idx starting at position pos."""
        if time.time() > time_limit:
            return False
        
        if r_idx >= n_rows:
            return True
        
        if pos >= L:
            # Row complete, move to next row
            return backtrack(r_idx + 1, 0, current_used_bits, depth)
        
        row = work[r_idx]
        
        if pos == 0:
            # First element is always 0
            return backtrack(r_idx, 1, current_used_bits, depth)
        
        # Try values from smallest to largest
        lower = row[pos - 1] + 1
        upper = target_scope if pos == L - 1 else row[pos + 1] - 1
        
        # For efficiency, try values in a randomized but bounded range
        if pos == L - 1:
            # Last element: try from upper down to lower for scope reduction
            for val in range(upper, lower - 1, -1):
                if val > target_scope:
                    continue
                if can_place(row, pos, val, current_used_bits):
                    row[pos] = val
                    # Compute new bits
                    new_bits = 0
                    for i in range(L):
                        if i < pos:
                            d = val - row[i]
                            if d <= target_scope:
                                new_bits |= (1 << d)
                        elif i > pos:
                            d = row[i] - val
                            if d <= target_scope:
                                new_bits |= (1 << d)
                    if backtrack(r_idx, pos + 1, current_used_bits | new_bits, depth + 1):
                        return True
        else:
            # Try a subset of values for efficiency
            max_attempts = 20
            attempts = 0
            for val in range(lower, upper):
                if attempts >= max_attempts:
                    break
                if val > target_scope:
                    break
                if can_place(row, pos, val, current_used_bits):
                    row[pos] = val
                    # Compute new bits
                    new_bits = 0
                    for i in range(L):
                        if i < pos:
                            d = val - row[i]
                            if d <= target_scope:
                                new_bits |= (1 << d)
                        elif i > pos:
                            d = row[i] - val
                            if d <= target_scope:
                                new_bits |= (1 << d)
                    if backtrack(r_idx, pos + 1, current_used_bits | new_bits, depth + 1):
                        return True
                attempts += 1
        
        return False
    
    # Try to reduce scope
    for attempt in range(50):
        if time.time() > time_limit:
            break
        
        # Reset work to best found so far
        work = [list(r) for r in best_rows]
        
        # Try backtracking from the last row
        success = backtrack(n_rows - 1, 0, 0)
        if success:
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
                target_scope = best_max - 1
            else:
                # No improvement, try a smaller target
                target_scope = max(112, target_scope - 1)
    
    return best_rows


def construct(n_rows, k):
    start_time = time.time()
    time_limit = start_time + 135

    # 1) Deterministic greedy baseline
    best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    best_max = _max_entry(best_rows)

    # 2) Many randomized greedy restarts with varied parameters
    rng = random.Random(42)
    greedy_deadline = time_limit - 60
    trial = 0
    while time.time() < greedy_deadline:
        trial += 1
        skip_p = rng.random() * 0.12
        jump_p = rng.random() * 0.05
        shuffle = rng.random() < 0.5
        bias_p = rng.random() * 0.15
        bias_range = rng.randint(2, 8)
        rows = _greedy_variant(n_rows, k, rng, skip_p=skip_p, jump_p=jump_p,
                               shuffle_rows=shuffle, bias_p=bias_p, bias_range=bias_range)
        if rows is not None and _check_valid(rows, k):
            rmax = _max_entry(rows)
            if rmax < best_max:
                best_rows = rows
                best_max = rmax

    # 3) Local compaction
    compacted = _local_compact(best_rows, n_rows, k, time_limit - 30)
    if _check_valid(compacted, k):
        cmax = _max_entry(compacted)
        if cmax < best_max:
            best_rows = compacted
            best_max = cmax

    # 4) Bitset backtracking for targeted scope reduction
    if time.time() < time_limit - 15:
        bt_result = _bitset_backtrack(best_rows, n_rows, k, rng, time_limit - 15)
        if _check_valid(bt_result, k):
            btmax = _max_entry(bt_result)
            if btmax < best_max:
                best_rows = bt_result
                best_max = btmax

    # 5) Simulated annealing with remaining time
    if time.time() < time_limit - 5:
        sa_result = _simulated_annealing(best_rows, n_rows, k, rng, time_limit - 5)
        if _check_valid(sa_result, k):
            smax = _max_entry(sa_result)
            if smax < best_max:
                best_rows = sa_result
                best_max = smax

    # 6) Final validation
    if not _check_valid(best_rows, k):
        best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))

    return best_rows
