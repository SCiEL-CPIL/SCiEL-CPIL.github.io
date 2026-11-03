"""Difference Triangle Set (4,7) solver.

Strategy: model as discrete tomography by enumerating valid row patterns
(sets of k+1 increasing integers starting at 0 whose internal differences
are all distinct) up to a scope bound, then select n_rows disjoint patterns
via a custom branch-and-bound ILP-style search.
Also includes greedy construction, local compaction, and simulated annealing.
"""
import time
import random
import sys


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


def _diffs_of_row(row):
    L = len(row)
    s = set()
    for i in range(L):
        for j in range(i + 1, L):
            s.add(row[j] - row[i])
    return s


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
    """Greedy with configurable parameters."""
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

    for attempt in range(80):
        strategy = rng.random()
        new_row = [0]
        x = 0
        valid = True

        if strategy < 0.4:
            while len(new_row) < L:
                x += 1
                if rng.random() < 0.05:
                    x += rng.randint(1, 3)
                new_diffs = {x - a for a in new_row}
                if new_diffs & used_set:
                    continue
                new_row.append(x)
                if x > best_new_max:
                    valid = False
                    break
        elif strategy < 0.7:
            while len(new_row) < L:
                x += rng.randint(1, 4)
                new_diffs = {x - a for a in new_row}
                if new_diffs & used_set:
                    continue
                new_row.append(x)
                if x > best_new_max:
                    valid = False
                    break
        else:
            while len(new_row) < L:
                x += 1
                if rng.random() < 0.15:
                    x += rng.randint(2, 5)
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


def _enumerate_valid_rows(k, max_scope, deadline):
    """Enumerate all valid row patterns (sets of k+1 increasing ints starting at 0)
    whose internal differences are all distinct, with max entry <= max_scope."""
    L = k + 1
    valid_rows = []
    
    def backtrack(current_row, current_diffs, x):
        if time.time() > deadline:
            return
        if len(current_row) == L:
            valid_rows.append(list(current_row))
            return
        # Try next possible values
        for next_x in range(x + 1, max_scope + 1):
            new_diffs = set(current_diffs)
            valid = True
            for a in current_row:
                d = next_x - a
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                current_row.append(next_x)
                backtrack(current_row, new_diffs, next_x)
                current_row.pop()
            if time.time() > deadline:
                return
    
    backtrack([0], set(), 0)
    return valid_rows


def _pattern_selection_search(valid_rows, n_rows, k, rng, time_limit, max_bound):
    """Select n_rows disjoint patterns from valid_rows to minimize max entry.
    Uses branch-and-bound with forward checking on the difference set."""
    L = k + 1
    num_diffs_per_row = L * (L - 1) // 2
    total_diffs_needed = n_rows * num_diffs_per_row
    
    # Sort patterns by max entry (ascending) for early pruning
    indexed_rows = [(row, row[-1]) for row in valid_rows if row[-1] <= max_bound]
    indexed_rows.sort(key=lambda x: x[1])
    
    # Precompute difference sets for each pattern
    pattern_diffs = []
    for row, mx in indexed_rows:
        diffs = set()
        for i in range(L):
            for j in range(i + 1, L):
                diffs.add(row[j] - row[i])
        pattern_diffs.append(diffs)
    
    best_solution = None
    best_max = max_bound
    deadline = time.time() + time_limit
    
    def select(n_selected, used_diffs, start_idx, current_max, selected_rows):
        nonlocal best_solution, best_max
        if time.time() > deadline:
            return False
        if n_selected == n_rows:
            if current_max < best_max:
                best_max = current_max
                best_solution = [list(r) for r in selected_rows]
            return True
        
        remaining_slots = n_rows - n_selected
        # Pruning: even if we pick the smallest remaining, can we fit?
        for idx in range(start_idx, len(indexed_rows)):
            row, mx = indexed_rows[idx]
            if mx >= current_max:
                break
            diffs = pattern_diffs[idx]
            if diffs & used_diffs:
                continue
            selected_rows.append(row)
            if select(n_selected + 1, used_diffs | diffs, idx + 1, max(current_max, mx), selected_rows):
                return True
            selected_rows.pop()
        return False
    
    # Try to find a solution with max < max_bound
    success = select(0, set(), 0, max_bound + 1, [])
    if success and best_solution is not None:
        return best_solution
    return None


def _simulated_annealing(rows, n_rows, k, rng, time_limit):
    """Simulated annealing to improve the solution."""
    work = [list(r) for r in rows]
    L = k + 1
    used_set = _build_used_set(work, k)
    current_max = _max_entry(work)
    best_rows = [list(r) for r in work]
    best_max = current_max

    start_temp = max(1, current_max // 8)
    end_temp = 0.05
    cooling = 0.9995

    t = start_temp
    stagnation_counter = 0
    max_stagnation = 2000

    while time.time() < time_limit and t > end_temp:
        move_type = rng.random()

        if move_type < 0.35:
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

        elif move_type < 0.65:
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            upper_limit = old_val + rng.randint(1, max(2, current_max // 3))
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
                for delta in range(1, 5):
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

        elif move_type < 0.80:
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

        elif move_type < 0.90:
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
            r1 = rng.randint(0, n_rows - 1)
            r2 = rng.randint(0, n_rows - 1)
            if r1 != r2:
                p1 = rng.randint(1, L - 1)
                p2 = rng.randint(1, L - 1)
                row1 = work[r1]
                row2 = work[r2]
                v1 = row1[p1]
                v2 = row2[p2]

                old_diffs1 = _get_diffs_for_pos(row1, p1, L)
                old_diffs2 = _get_diffs_for_pos(row2, p2, L)

                if p1 > 0 and v2 <= row1[p1 - 1]:
                    pass
                elif p1 < L - 1 and v2 >= row1[p1 + 1]:
                    pass
                else:
                    if p2 > 0 and v1 <= row2[p2 - 1]:
                        pass
                    elif p2 < L - 1 and v1 >= row2[p2 + 1]:
                        pass
                    else:
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

        if stagnation_counter > max_stagnation:
            fresh = _greedy_variant(n_rows, k, rng, skip_p=rng.random() * 0.15,
                                    jump_p=rng.random() * 0.08, shuffle_rows=True,
                                    bias_p=rng.random() * 0.3, bias_range=rng.randint(3, 8))
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
    time_limit = start_time + 140

    # 1) Deterministic greedy baseline
    best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    best_max = _max_entry(best_rows)

    # 2) Many randomized greedy restarts
    rng = random.Random(42)
    greedy_deadline = time_limit - 80
    trial = 0
    while time.time() < greedy_deadline:
        trial += 1
        skip_p = rng.random() * 0.15
        jump_p = rng.random() * 0.08
        shuffle = rng.random() < 0.6
        bias_p = rng.random() * 0.4
        bias_range = rng.randint(3, 8)
        rows = _greedy_variant(n_rows, k, rng, skip_p=skip_p, jump_p=jump_p,
                               shuffle_rows=shuffle, bias_p=bias_p, bias_range=bias_range)
        if rows is not None and _check_valid(rows, k):
            rmax = _max_entry(rows)
            if rmax < best_max:
                best_rows = rows
                best_max = rmax

    # 3) Pattern enumeration + selection search (discrete tomography approach)
    enum_deadline = time_limit - 50
    if time.time() < enum_deadline:
        # Enumerate valid row patterns up to a bound slightly below current best
        enum_bound = best_max
        enum_time_left = enum_deadline - time.time() - 5
        if enum_time_left > 3:
            valid_patterns = _enumerate_valid_rows(k, enum_bound, enum_deadline - 5)
            if len(valid_patterns) > 0:
                sel_time_left = enum_deadline - time.time() - 2
                if sel_time_left > 2:
                    sel_result = _pattern_selection_search(valid_patterns, n_rows, k, rng, sel_time_left, best_max)
                    if sel_result is not None and _check_valid(sel_result, k):
                        smax = _max_entry(sel_result)
                        if smax < best_max:
                            best_rows = sel_result
                            best_max = smax

    # 4) Local compaction
    compact_deadline = time_limit - 20
    if time.time() < compact_deadline:
        compacted = _local_compact(best_rows, n_rows, k, compact_deadline)
        if _check_valid(compacted, k):
            cmax = _max_entry(compacted)
            if cmax < best_max:
                best_rows = compacted
                best_max = cmax

    # 5) Simulated annealing
    if time.time() < time_limit - 3:
        sa_result = _simulated_annealing(best_rows, n_rows, k, rng, time_limit - 3)
        if _check_valid(sa_result, k):
            smax = _max_entry(sa_result)
            if smax < best_max:
                best_rows = sa_result
                best_max = smax

    # 6) Final validation
    if not _check_valid(best_rows, k):
        best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))

    return best_rows
