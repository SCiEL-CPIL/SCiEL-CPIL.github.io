"""
Difference Triangle Set (4,7) solver – structural rewrite.

New design:
1. Fast, vectorised greedy baseline with randomised parameters.
2. Iterated Local Search (ILS) with two neighbourhood operators:
   (a) element-move (shift an element to a nearby position)
   (b) row-rebuild (rebuild one row greedily under the current difference
       occupancy, then compact)
3. A difference-packing search: treat the 144 required differences (4 rows
   × 36) as objects to pack into the smallest possible range [1, M].
   We use a backtracking search that assigns the most constrained
   differences first, with forward checking.
4. A final polish pass using aggressive element shifting.

The key structural change is in step 3: instead of building rows
element-by-element, we assign difference slots to concrete values,
which allows us to prune the search based on the difference-set
occupancy rather than row-level feasibility.
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


def _all_diffs(rows, k):
    """Return the set of all differences in the rows."""
    used = set()
    L = k + 1
    for row in rows:
        for i in range(L):
            for j in range(i + 1, L):
                used.add(row[j] - row[i])
    return used


def _greedy_row(used, k, rng, skip_p=0.05, jump_p=0.03, bias_p=0.0, bias_range=5):
    """Build a single row greedily, avoiding used differences."""
    row = [0]
    x = 0
    L = k + 1
    while len(row) < L:
        x += 1
        if rng.random() < jump_p:
            x += rng.randint(1, 4)
        if rng.random() < skip_p:
            continue
        if bias_p > 0 and rng.random() < bias_p:
            x += rng.randint(1, bias_range)
        new_diffs = {x - a for a in row}
        if new_diffs & used:
            continue
        row.append(x)
        used |= new_diffs
    return row


def _greedy_construct(n_rows, k, rng, skip_p=0.05, jump_p=0.03, bias_p=0.0, bias_range=5, shuffle_rows=False):
    """Greedy construction of all rows."""
    used = set()
    rows = []
    order = list(range(n_rows))
    if shuffle_rows:
        rng.shuffle(order)
    for r_idx in order:
        row = _greedy_row(used, k, rng, skip_p, jump_p, bias_p, bias_range)
        rows.append(row)
    # Reorder rows back
    result = [None] * n_rows
    for r_idx in order:
        result[r_idx] = rows[r_idx]
    return result


def _compact_element(row, pos, used, L):
    """Try to compact row[pos] to its smallest feasible value."""
    old_val = row[pos]
    if pos == 0:
        return False
    lower = row[pos - 1] + 1
    if lower >= old_val:
        return False
    old_diffs = set()
    for i in range(L):
        if i < pos:
            old_diffs.add(row[pos] - row[i])
        elif i > pos:
            old_diffs.add(row[i] - row[pos])
    other = used - old_diffs
    for new_val in range(lower, old_val):
        if pos < L - 1 and new_val >= row[pos + 1]:
            break
        new_diffs = set()
        for i in range(L):
            if i < pos:
                new_diffs.add(new_val - row[i])
            elif i > pos:
                new_diffs.add(row[i] - new_val)
        if new_diffs & other:
            continue
        # Apply change
        for d in old_diffs:
            used.discard(d)
        for d in new_diffs:
            used.add(d)
        row[pos] = new_val
        return True
    return False


def _compact_row(row, used, L):
    """Compact all elements in a row."""
    changed = False
    for pos in range(L - 1, 0, -1):
        if _compact_element(row, pos, used, L):
            changed = True
    return changed


def _local_search(rows, n_rows, k, rng, time_limit, max_iterations=100000):
    """Iterated local search: repeatedly compact and rebuild."""
    work = [list(r) for r in rows]
    L = k + 1
    used = _all_diffs(work, k)
    best_rows = [list(r) for r in work]
    best_max = _max_entry(work)
    
    iterations = 0
    while time.time() < time_limit and iterations < max_iterations:
        iterations += 1
        # Try to improve by compacting a random element
        r_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, L - 1)
        if _compact_element(work[r_idx], pos, used, L):
            # Full recompact after a successful move
            for r in range(n_rows):
                _compact_row(work[r], used, L)
            cur_max = _max_entry(work)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [list(r) for r in work]
        
        # Occasionally rebuild a row
        if rng.random() < 0.05:
            r_idx = rng.randint(0, n_rows - 1)
            old_row = work[r_idx]
            old_diffs = set()
            for i in range(L):
                for j in range(i + 1, L):
                    old_diffs.add(old_row[j] - old_row[i])
            for d in old_diffs:
                used.discard(d)
            
            new_row = _greedy_row(used, k, rng)
            if new_row[-1] < best_max or rng.random() < 0.1:
                for i in range(L):
                    for j in range(i + 1, L):
                        used.add(new_row[j] - new_row[i])
                work[r_idx] = new_row
                for r in range(n_rows):
                    _compact_row(work[r], used, L)
                cur_max = _max_entry(work)
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [list(r) for r in work]
            else:
                # Revert
                for d in used:
                    pass
                # Rebuild from scratch is complex; just keep old row
                for i in range(L):
                    for j in range(i + 1, L):
                        used.discard(new_row[j] - new_row[i])
                for i in range(L):
                    for j in range(i + 1, L):
                        used.add(old_row[j] - old_row[i])
                work[r_idx] = old_row
    
    return best_rows


def _difference_packing_search(n_rows, k, rng, time_limit, max_bound):
    """
    Difference-packing search: assign the 144 required differences
    to values in [1, M], subject to the constraint that for each row,
    the differences must be consistent with some set of k+1 positions.
    
    We use a backtracking search that assigns differences to (row, i, j)
    slots, with forward checking on the difference set.
    """
    L = k + 1
    num_diffs_per_row = L * (L - 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    start_time = time.time()
    best_solution = None
    best_max = max_bound
    
    # Precompute all (i, j) pairs for a row
    pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]
    
    def try_assign(rows, used_diffs, deadline):
        """Try to complete the assignment using forward checking."""
        nonlocal best_solution, best_max
        
        if time.time() > deadline:
            return False
        
        # Find the most constrained unassigned position
        # A position is "constrained" if it has few feasible values
        min_feasible = float('inf')
        best_r, best_pos = -1, -1
        
        for r in range(n_rows):
            for pos in range(1, L):
                if rows[r][pos] == -1:  # unassigned
                    # Count feasible values
                    lower = rows[r][pos - 1] + 1 if pos > 0 and rows[r][pos - 1] != -1 else 1
                    upper = rows[r][pos + 1] - 1 if pos < L - 1 and rows[r][pos + 1] != -1 else max_bound - 1
                    if lower > upper:
                        return False
                    
                    count = 0
                    for val in range(lower, min(upper + 1, max_bound)):
                        ok = True
                        for i in range(L):
                            if i == pos or rows[r][i] == -1:
                                continue
                            d = abs(val - rows[r][i])
                            if d in used_diffs:
                                ok = False
                                break
                        if ok:
                            count += 1
                            if count >= 20:  # Early termination for speed
                                break
                    
                    if count == 0:
                        return False
                    if count < min_feasible:
                        min_feasible = count
                        best_r, best_pos = r, pos
                        if count == 1:
                            break
            if min_feasible == 1:
                break
        
        if best_r == -1:
            # All assigned
            row_max = max(max(r) for r in rows)
            if row_max < best_max:
                best_max = row_max
                best_solution = [list(r) for r in rows]
            return True
        
        # Assign best_r, best_pos
        lower = rows[best_r][best_pos - 1] + 1 if best_pos > 0 and rows[best_r][best_pos - 1] != -1 else 1
        upper = rows[best_r][best_pos + 1] - 1 if best_pos < L - 1 and rows[best_r][best_pos + 1] != -1 else max_bound - 1
        
        if lower > upper:
            return False
        
        candidates = []
        for val in range(lower, min(upper + 1, max_bound)):
            ok = True
            for i in range(L):
                if i == best_pos or rows[best_r][i] == -1:
                    continue
                d = abs(val - rows[best_r][i])
                if d in used_diffs:
                    ok = False
                    break
            if ok:
                candidates.append(val)
        
        if not candidates:
            return False
        
        # Shuffle for diversity
        rng.shuffle(candidates)
        
        for val in candidates:
            if time.time() > deadline:
                return False
            
            rows[best_r][best_pos] = val
            new_diffs = set()
            for i in range(L):
                if i != best_pos and rows[best_r][i] != -1:
                    d = abs(val - rows[best_r][i])
                    new_diffs.add(d)
            
            if not (new_diffs & used_diffs):
                used_diffs.update(new_diffs)
                if try_assign(rows, used_diffs, deadline):
                    return True
                used_diffs.difference_update(new_diffs)
            
            rows[best_r][best_pos] = -1
        
        return False
    
    # Multiple restarts
    for restart in range(50):
        if time.time() > time_limit:
            break
        
        rows = [[-1] * L for _ in range(n_rows)]
        for r in range(n_rows):
            rows[r][0] = 0
        used_diffs = set()
        
        effective_bound = best_max - 1 if best_max < 1000 else max_bound
        
        if try_assign(rows, used_diffs, time_limit):
            if best_solution is not None:
                return best_solution
    
    return best_solution


def _simulated_annealing(rows, n_rows, k, rng, time_limit):
    """Simulated annealing to improve the solution."""
    work = [list(r) for r in rows]
    L = k + 1
    used = _all_diffs(work, k)
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
            # Check if new_val is feasible
            old_diffs = set()
            for i in range(L):
                if i < pos:
                    old_diffs.add(row[pos] - row[i])
                elif i > pos:
                    old_diffs.add(row[i] - row[pos])
            new_diffs = set()
            for i in range(L):
                if i < pos:
                    new_diffs.add(new_val - row[i])
                elif i > pos:
                    new_diffs.add(row[i] - new_val)
            other = used - old_diffs
            if not (new_diffs & other):
                for d in old_diffs:
                    used.discard(d)
                for d in new_diffs:
                    used.add(d)
                row[pos] = new_val
                _compact_row(row, used, L)
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]
                    current_max = best_max
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

        elif move_type < 0.65:
            # Move 2: Increase a random element (perturbation), then recompact
            r_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, L - 1)
            row = work[r_idx]
            old_val = row[pos]
            upper_limit = old_val + rng.randint(1, max(2, current_max // 3))
            new_val = rng.randint(old_val + 1, upper_limit)
            if new_val <= old_val:
                t *= cooling
                continue
            old_diffs = set()
            for i in range(L):
                if i < pos:
                    old_diffs.add(row[pos] - row[i])
                elif i > pos:
                    old_diffs.add(row[i] - row[pos])
            new_diffs = set()
            for i in range(L):
                if i < pos:
                    new_diffs.add(new_val - row[i])
                elif i > pos:
                    new_diffs.add(row[i] - new_val)
            other = used - old_diffs
            if not (new_diffs & other):
                for d in old_diffs:
                    used.discard(d)
                for d in new_diffs:
                    used.add(d)
                row[pos] = new_val
                _compact_row(row, used, L)
                new_max = _max_entry(work)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in work]
                    current_max = best_max
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

        elif move_type < 0.80:
            # Move 3: Rebuild a random row
            r_idx = rng.randint(0, n_rows - 1)
            old_row = work[r_idx]
            old_diffs = set()
            for i in range(L):
                for j in range(i + 1, L):
                    old_diffs.add(old_row[j] - old_row[i])
            for d in old_diffs:
                used.discard(d)
            
            new_row = _greedy_row(used, k, rng)
            for i in range(L):
                for j in range(i + 1, L):
                    used.add(new_row[j] - new_row[i])
            work[r_idx] = new_row
            for r in range(n_rows):
                _compact_row(work[r], used, L)
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
                current_max = best_max
                stagnation_counter = 0
            else:
                stagnation_counter += 1

        elif move_type < 0.90:
            # Move 4: Full recompact of all rows
            for r_idx in range(n_rows):
                _compact_row(work[r_idx], used, L)
            new_max = _max_entry(work)
            if new_max < best_max:
                best_max = new_max
                best_rows = [list(r) for r in work]
                current_max = best_max
                stagnation_counter = 0
            else:
                stagnation_counter += 1

        else:
            # Move 5: Swap two elements between different rows
            r1 = rng.randint(0, n_rows - 1)
            r2 = rng.randint(0, n_rows - 1)
            if r1 != r2:
                p1 = rng.randint(1, L - 1)
                p2 = rng.randint(1, L - 1)
                row1 = work[r1]
                row2 = work[r2]
                v1 = row1[p1]
                v2 = row2[p2]

                # Check if swap is feasible
                old_diffs1 = set()
                for i in range(L):
                    if i < p1:
                        old_diffs1.add(row1[p1] - row1[i])
                    elif i > p1:
                        old_diffs1.add(row1[i] - row1[p1])
                old_diffs2 = set()
                for i in range(L):
                    if i < p2:
                        old_diffs2.add(row2[p2] - row2[i])
                    elif i > p2:
                        old_diffs2.add(row2[i] - row2[p2])
                
                # Check if v2 is feasible in row1 at p1
                new_diffs1 = set()
                for i in range(L):
                    if i < p1:
                        new_diffs1.add(v2 - row1[i])
                    elif i > p1:
                        new_diffs1.add(row1[i] - v2)
                
                # Check if v1 is feasible in row2 at p2
                new_diffs2 = set()
                for i in range(L):
                    if i < p2:
                        new_diffs2.add(v1 - row2[i])
                    elif i > p2:
                        new_diffs2.add(row2[i] - v1)
                
                # Check ordering constraints
                if (p1 > 0 and v2 <= row1[p1 - 1]) or (p1 < L - 1 and v2 >= row1[p1 + 1]):
                    pass
                elif (p2 > 0 and v1 <= row2[p2 - 1]) or (p2 < L - 1 and v1 >= row2[p2 + 1]):
                    pass
                else:
                    other = used - old_diffs1 - old_diffs2
                    if not (new_diffs1 & other) and not (new_diffs2 & other) and not (new_diffs1 & new_diffs2):
                        for d in old_diffs1:
                            used.discard(d)
                        for d in old_diffs2:
                            used.discard(d)
                        for d in new_diffs1:
                            used.add(d)
                        for d in new_diffs2:
                            used.add(d)
                        row1[p1] = v2
                        row2[p2] = v1
                        _compact_row(row1, used, L)
                        _compact_row(row2, used, L)
                        new_max = _max_entry(work)
                        if new_max < best_max:
                            best_max = new_max
                            best_rows = [list(r) for r in work]
                            current_max = best_max
                            stagnation_counter = 0
                        else:
                            stagnation_counter += 1

        if stagnation_counter > max_stagnation:
            # Restart with a fresh greedy solution
            fresh = _greedy_construct(n_rows, k, rng, skip_p=rng.random() * 0.15,
                                      jump_p=rng.random() * 0.08, shuffle_rows=True,
                                      bias_p=rng.random() * 0.3, bias_range=rng.randint(3, 8))
            if fresh is not None and _check_valid(fresh, k):
                fresh_max = _max_entry(fresh)
                if fresh_max < best_max:
                    best_max = fresh_max
                    best_rows = [list(r) for r in fresh]
                    current_max = best_max
                    work = [list(r) for r in fresh]
                    used = _all_diffs(work, k)
                    t = start_temp
                    stagnation_counter = 0

        t *= cooling

    return best_rows


def _gap_based_local_search(rows, n_rows, k, rng, time_limit):
    """
    Gap-based local search: represent each row by its k consecutive gaps.
    Perform gap-exchange moves between different rows to explore structural
    reconfigurations that single-row element shifts cannot access.
    """
    work = [list(r) for r in rows]
    L = k + 1
    used = _all_diffs(work, k)
    best_rows = [list(r) for r in work]
    best_max = _max_entry(work)

    def row_to_gaps(row):
        return [row[i+1] - row[i] for i in range(k)]

    def gaps_to_row(gaps):
        row = [0]
        x = 0
        for g in gaps:
            x += g
            row.append(x)
        return row

    # Precompute gaps for each row
    all_gaps = [row_to_gaps(row) for row in work]

    def check_and_update(r_idx, new_row):
        """Check if new_row is valid and update used set."""
        nonlocal used
        # Compute new differences
        new_diffs = set()
        for i in range(L):
            for j in range(i+1, L):
                new_diffs.add(new_row[j] - new_row[i])
        
        # Check for conflicts with other rows
        for other_r in range(n_rows):
            if other_r == r_idx:
                continue
            for i in range(L):
                for j in range(i+1, L):
                    d = work[other_r][j] - work[other_r][i]
                    if d in new_diffs:
                        return False
        return True

    iterations = 0
    while time.time() < time_limit and iterations < 50000:
        iterations += 1

        # Move 1: Gap exchange between two rows
        if rng.random() < 0.4:
            r1 = rng.randint(0, n_rows - 1)
            r2 = rng.randint(0, n_rows - 1)
            if r1 == r2:
                continue
            g1 = rng.randint(0, k - 1)
            g2 = rng.randint(0, k - 1)

            old_g1 = all_gaps[r1][g1]
            old_g2 = all_gaps[r2][g2]

            # Try swapping gaps
            all_gaps[r1][g1] = old_g2
            all_gaps[r2][g2] = old_g1

            new_row1 = gaps_to_row(all_gaps[r1])
            new_row2 = gaps_to_row(all_gaps[r2])

            # Check validity
            if new_row1[-1] >= 0 and new_row2[-1] >= 0:
                # Check that all elements are positive and increasing
                valid = True
                for row in [new_row1, new_row2]:
                    for i in range(1, L):
                        if row[i] <= row[i-1]:
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    # Check difference conflicts
                    if check_and_update(r1, new_row1) and check_and_update(r2, new_row2):
                        work[r1] = new_row1
                        work[r2] = new_row2
                        used = _all_diffs(work, k)
                        cur_max = _max_entry(work)
                        if cur_max < best_max:
                            best_max = cur_max
                            best_rows = [list(r) for r in work]
                        # Compact after successful move
                        for r in range(n_rows):
                            _compact_row(work[r], used, L)
                        cur_max = _max_entry(work)
                        if cur_max < best_max:
                            best_max = cur_max
                            best_rows = [list(r) for r in work]
                    else:
                        # Revert
                        all_gaps[r1][g1] = old_g1
                        all_gaps[r2][g2] = old_g2
                else:
                    all_gaps[r1][g1] = old_g1
                    all_gaps[r2][g2] = old_g2
            else:
                all_gaps[r1][g1] = old_g1
                all_gaps[r2][g2] = old_g2

        # Move 2: Gap perturbation (increase or decrease a gap)
        elif rng.random() < 0.7:
            r_idx = rng.randint(0, n_rows - 1)
            g_idx = rng.randint(0, k - 1)
            old_gap = all_gaps[r_idx][g_idx]

            if rng.random() < 0.5:
                # Try to decrease the gap
                new_gap = max(1, old_gap - rng.randint(1, 3))
            else:
                # Try to increase the gap
                new_gap = old_gap + rng.randint(1, 3)

            if new_gap == old_gap:
                continue

            all_gaps[r_idx][g_idx] = new_gap
            new_row = gaps_to_row(all_gaps[r_idx])

            # Check validity
            valid = True
            for i in range(1, L):
                if new_row[i] <= new_row[i-1]:
                    valid = False
                    break

            if valid and check_and_update(r_idx, new_row):
                work[r_idx] = new_row
                used = _all_diffs(work, k)
                _compact_row(work[r_idx], used, L)
                cur_max = _max_entry(work)
                if cur_max < best_max:
                    best_max = cur_max
                    best_rows = [list(r) for r in work]
            else:
                all_gaps[r_idx][g_idx] = old_gap

        # Move 3: Compact all rows
        else:
            for r in range(n_rows):
                _compact_row(work[r], used, L)
            cur_max = _max_entry(work)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [list(r) for r in work]

    return best_rows


def construct(n_rows, k):
    start_time = time.time()
    time_limit = start_time + 140  # Safety margin before 150s kill

    # 1) Deterministic greedy baseline
    best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))
    best_max = _max_entry(best_rows)

    # 2) Many randomized greedy restarts
    rng = random.Random(42)
    greedy_deadline = time_limit - 85
    trial = 0
    while time.time() < greedy_deadline:
        trial += 1
        skip_p = rng.random() * 0.15
        jump_p = rng.random() * 0.08
        shuffle = rng.random() < 0.6
        bias_p = rng.random() * 0.4
        bias_range = rng.randint(3, 8)
        rows = _greedy_construct(n_rows, k, rng, skip_p=skip_p, jump_p=jump_p,
                                 shuffle_rows=shuffle, bias_p=bias_p, bias_range=bias_range)
        if rows is not None and _check_valid(rows, k):
            rmax = _max_entry(rows)
            if rmax < best_max:
                best_rows = rows
                best_max = rmax

    # 3) Difference-based CSP search
    csp_deadline = time_limit - 50
    if time.time() < csp_deadline:
        csp_result = _difference_packing_search(n_rows, k, rng, csp_deadline - time.time(), max_bound=best_max)
        if csp_result is not None and _check_valid(csp_result, k):
            cmax = _max_entry(csp_result)
            if cmax < best_max:
                best_rows = csp_result
                best_max = cmax

    # 4) Local search
    ls_deadline = time_limit - 25
    if time.time() < ls_deadline:
        ls_result = _local_search(best_rows, n_rows, k, rng, ls_deadline - time.time())
        if _check_valid(ls_result, k):
            lmax = _max_entry(ls_result)
            if lmax < best_max:
                best_rows = ls_result
                best_max = lmax

    # 5) Gap-based local search (new structural approach)
    gap_deadline = time_limit - 10
    if time.time() < gap_deadline:
        gap_result = _gap_based_local_search(best_rows, n_rows, k, rng, gap_deadline - time.time())
        if _check_valid(gap_result, k):
            gmax = _max_entry(gap_result)
            if gmax < best_max:
                best_rows = gap_result
                best_max = gmax

    # 6) Simulated annealing
    if time.time() < time_limit - 3:
        sa_result = _simulated_annealing(best_rows, n_rows, k, rng, time_limit - 3)
        if _check_valid(sa_result, k):
            smax = _max_entry(sa_result)
            if smax < best_max:
                best_rows = sa_result
                best_max = smax

    # 7) Final validation
    if not _check_valid(best_rows, k):
        best_rows = _greedy_construct(n_rows, k, rng=random.Random(0))

    return best_rows
