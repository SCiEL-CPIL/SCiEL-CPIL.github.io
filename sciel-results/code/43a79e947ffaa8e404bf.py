"""Difference Triangle Set (4,7) solver.

Combined strategy:
- Fast greedy baseline + many randomized greedy variants (with row-order shuffling).
- Efficient local compaction (right-to-left, set-based difference checks).
- Time-bounded simulated annealing with:
    * single-element decrease (compaction) moves
    * random element perturbations + recompact
    * full-row rebuilds (multiple attempts, bounded by current max)
    * full recompact passes
    * pair-swap moves between rows
- Safety margin before the 150 s wall-clock limit.
"""
import time
import random
import numpy as np


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


def _greedy_variant(n_rows, k, rng, skip_p=0.05, jump_p=0.03, shuffle_rows=False, bias_p=0.0, bias_range=5):
    """Greedy with configurable skip and jump probabilities, optional row-order shuffle, and bias."""
    used = set()
    rows = []
    # Optionally shuffle the order in which rows are built
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
            # Bias: sometimes try a larger jump
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
    """Rebuild a row from scratch, trying to keep its max below max_val."""
    row = work[r_idx]
    old_diffs = set()
    for i in range(L):
        for j in range(i + 1, L):
            old_diffs.add(row[j] - row[i])
    for d in old_diffs:
        used_set.discard(d)

    best_new_row = None
    best_new_max = max_val + 1

    # Try multiple strategies
    for attempt in range(80):
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
        elif strategy < 0.85:
            # Conservative with occasional jumps
            while len(new_row) < L:
                x += 1
                if rng.random() < 0.20:
                    x += rng.randint(2, 6)
                new_diffs = {x - a for a in new_row}
                if new_diffs & used_set:
                    continue
                new_row.append(x)
                if x > best_new_max:
                    valid = False
                    break
        else:
            # Randomized step sizes
            while len(new_row) < L:
                x += rng.randint(1, 4)
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
        # Add new row's diffs to used_set
        for i in range(L):
            for j in range(i + 1, L):
                used_set.add(best_new_row[j] - best_new_row[i])
        work[r_idx] = best_new_row
        return True
    else:
        # Restore old row
        for d in old_diffs:
            used_set.add(d)
        work[r_idx] = row
        return False


def _simulated_annealing(rows, n_rows, k, rng, time_limit):
    """Simulated annealing to improve the solution."""
    work = [list(r) for r in rows]
    L = k + 1
    used_set = _build_used_set(work, k)
    current_max = _max_entry(work)
    best_rows = [list(r) for r in work]
    best_max = current_max

    start_temp = max(1, current_max // 6)
    end_temp = 0.05
    cooling = 0.9997

    t = start_temp
    stagnation_counter = 0
    max_stagnation = 1500

    while time.time() < time_limit and t > end_temp:
        move_type = rng.random()

        if move_type < 0.30:
            # Move 1: Try to decrease a random element
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
                    current_max = best_max
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

        elif move_type < 0.55:
            # Move 2: Increase a random element (perturbation), then recompact
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            upper_limit = old_val + rng.randint(1, max(3, current_max // 4))
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
                    current_max = best_max
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            else:
                # Try smaller increments
                for delta in range(1, 6):
                    candidate = old_val + delta
                    if _can_set(row, pos, candidate, used_set, L):
                        _apply_set(row, pos, candidate, used_set, L)
                        _fast_compact_row(row, used_set, L)
                        new_max = _max_entry(work)
                        if new_max < best_max:
                            best_max = new_max
                            best_rows = [list(r) for r in work]
                            current_max = best_max
                            stagnation_counter = 0
                        break

        elif move_type < 0.70:
            # Move 3: Rebuild a random row
            r_idx = rng.randint(0, n_rows - 1)
            _rebuild_row(r_idx, work, used_set, L, rng, current_max)
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
                current_max = best_max
                stagnation_counter = 0
            else:
                stagnation_counter += 1

        elif move_type < 0.80:
            # Move 4: Full recompact of all rows
            for r_idx in range(n_rows):
                _fast_compact_row(work[r_idx], used_set, L)
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
                current_max = best_max
                stagnation_counter = 0
            else:
                stagnation_counter += 1

        else:
            # Move 5: Swap two elements between different rows (if valid)
            r1 = rng.randint(0, n_rows - 1)
            r2 = rng.randint(0, n_rows - 1)
            if r1 != r2:
                p1 = rng.randint(1, L - 1)
                p2 = rng.randint(1, L - 1)
                row1 = work[r1]
                row2 = work[r2]
                v1 = row1[p1]
                v2 = row2[p2]
                
                # Check bounds
                bounds_ok = True
                if p1 > 0 and v2 <= row1[p1-1]:
                    bounds_ok = False
                if p1 < L-1 and v2 >= row1[p1+1]:
                    bounds_ok = False
                if p2 > 0 and v1 <= row2[p2-1]:
                    bounds_ok = False
                if p2 < L-1 and v1 >= row2[p2+1]:
                    bounds_ok = False
                
                if bounds_ok:
                    old_diffs1 = _get_diffs_for_pos(row1, p1, L)
                    old_diffs2 = _get_diffs_for_pos(row2, p2, L)
                    
                    new_diffs1 = set()
                    for i in range(L):
                        if i < p1:
                            new_diffs1.add(v2 - row1[i])
                        elif i > p1:
                            new_diffs1.add(row1[i] - v2)
                    
                    new_diffs2 = set()
                    for i in range(L):
                        if i < p2:
                            new_diffs2.add(v1 - row2[i])
                        elif i > p2:
                            new_diffs2.add(row2[i] - v1)
                    
                    other = used_set - old_diffs1 - old_diffs2
                    if not (new_diffs1 & other) and not (new_diffs2 & other) and not (new_diffs1 & new_diffs2):
                        # Apply swap
                        for d in old_diffs1:
                            used_set.discard(d)
                        for d in old_diffs2:
                            used_set.discard(d)
                        for d in new_diffs1:
                            used_set.add(d)
                        for d in new_diffs2:
                            used_set.add(d)
                        row1[p1] = v2
                        row2[p2] = v1
                        _fast_compact_row(row1, used_set, L)
                        _fast_compact_row(row2, used_set, L)
                        new_max = _max_entry(work)
                        if new_max < best_max:
                            best_max = new_max
                            best_rows = [list(r) for r in work]
                            current_max = best_max
                            stagnation_counter = 0
                        else:
                            stagnation_counter += 1
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1

        # If stuck for too long, restart with a fresh greedy
        if stagnation_counter > max_stagnation:
            fresh = _greedy_variant(n_rows, k, rng, skip_p=rng.random()*0.18, 
                                    jump_p=rng.random()*0.10, shuffle_rows=True,
                                    bias_p=rng.random()*0.35, bias_range=rng.randint(3,10))
            if fresh is not None and _check_valid(fresh, k):
                fresh_max = _max_entry(fresh)
                if fresh_max < best_max:
                    best_max = fresh_max
                    best_rows = [list(r) for r in fresh]
                    current_max = best_max
                    work = [list(r) for r in fresh]
                    used_set = _build_used_set(work, k)
                    t = start_temp
                    stagnation_counter = 0
            else:
                # Try a different restart strategy
                fresh = _greedy_variant(n_rows, k, rng, skip_p=0.0, 
                                        jump_p=0.0, shuffle_rows=False,
                                        bias_p=0.0, bias_range=0)
                if fresh is not None and _check_valid(fresh, k):
                    fresh_max = _max_entry(fresh)
                    if fresh_max < best_max:
                        best_max = fresh_max
                        best_rows = [list(r) for r in fresh]
                        current_max = best_max
                        work = [list(r) for r in fresh]
                        used_set = _build_used_set(work, k)
                        t = start_temp
                        stagnation_counter = 0

        t *= cooling

    return best_rows


def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: stop searching at 142 s to leave room for final validation
    time_limit = start_time + 142

    # 1) Deterministic greedy baseline
    best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    best_max = _max_entry(best_rows)

    # 2) Many randomized greedy restarts with varied parameters
    rng = random.Random(42)
    greedy_deadline = time_limit - 55
    trial = 0
    while time.time() < greedy_deadline:
        trial += 1
        # More diverse parameter ranges
        skip_p = rng.random() * 0.18
        jump_p = rng.random() * 0.10
        shuffle = rng.random() < 0.6
        bias_p = rng.random() * 0.45
        bias_range = rng.randint(3, 10)
        rows = _greedy_variant(n_rows, k, rng, skip_p=skip_p, jump_p=jump_p,
                               shuffle_rows=shuffle, bias_p=bias_p, bias_range=bias_range)
        if rows is not None and _check_valid(rows, k):
            rmax = _max_entry(rows)
            if rmax < best_max:
                best_rows = rows
                best_max = rmax

    # 3) Local compaction
    compacted = _local_compact(best_rows, n_rows, k, time_limit - 25)
    if _check_valid(compacted, k):
        cmax = _max_entry(compacted)
        if cmax < best_max:
            best_rows = compacted
            best_max = cmax

    # 4) Simulated annealing with remaining time
    if time.time() < time_limit - 2:
        sa_result = _simulated_annealing(best_rows, n_rows, k, rng, time_limit - 2)
        if _check_valid(sa_result, k):
            smax = _max_entry(sa_result)
            if smax < best_max:
                best_rows = sa_result
                best_max = smax

    # 5) Final validation
    if not _check_valid(best_rows, k):
        best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))

    return best_rows
