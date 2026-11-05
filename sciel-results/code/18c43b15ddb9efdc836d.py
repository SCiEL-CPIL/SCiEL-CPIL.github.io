# EVOLVE-BLOCK-START
import sys
import time
import random
import math

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.

    Hybrid approach:
    1. Multi-start randomized greedy for a valid baseline.
    2. Row-rebuild local search with difference-pressure candidate selection
       and occasional multi-row rebuilds to escape local minima.
    3. Element-level simulated annealing polish in the final phase.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)

    start_time = time.time()
    deadline = start_time + 117.0  # Safety margin before 120s limit

    def time_up():
        return time.time() > deadline

    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed

    def get_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def get_max(rows):
        return max(max(r) for r in rows)

    # ---- 1. Multi-start Greedy with difference-pressure candidate selection ----
    def greedy_construct(seed):
        rng = random.Random(seed)
        used = set()
        rows = []

        for _ in range(n_rows):
            row = [0]
            local_used = set()

            for pos in range(k):
                last = row[-1]
                remaining = k - pos - 1
                # We don't know the final max yet, so search a reasonable range
                # Collect candidates and score them by difference-pressure
                candidates = []
                x = last + 1
                # Search up to a generous bound; for small k,n_rows this is fine
                max_search = last + 200
                while x <= max_search:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & local_used) and not (new_diffs & used):
                        # Score: prefer smaller x and smaller max new diff
                        max_new_diff = max(new_diffs)
                        score = x + 0.3 * max_new_diff
                        candidates.append((score, x, new_diffs))
                    x += 1

                    if len(candidates) >= 15:
                        break

                if not candidates:
                    return None

                # Pick from top candidates with some randomness
                candidates.sort(key=lambda c: c[0])
                top_n = min(5, len(candidates))
                # Weight toward smaller scores
                weights = [1.0 / (i + 1) for i in range(top_n)]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                chosen_idx = 0
                for i in range(top_n):
                    cum += weights[i]
                    if r <= cum:
                        chosen_idx = i
                        break

                _, x, new_diffs = candidates[chosen_idx]
                row.append(x)
                local_used.update(new_diffs)

            rows.append(row)
            used.update(local_used)

        return rows, used

    # Initial best solution
    best_rows = None
    best_max = float('inf')
    current_rows = None
    current_used = None

    for seed in range(20):
        if time_up():
            break
        res = greedy_construct(seed)
        if res:
            rows, used = res
            m = get_max(rows)
            if m < best_max:
                best_max = m
                best_rows = [r[:] for r in rows]
                current_rows = [r[:] for r in rows]
                current_used = set(used)

    if best_rows is None:
        # Fallback: deterministic greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            local_used = set()
            for _ in range(k):
                last = row[-1]
                x = last + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and not (new_diffs & local_used):
                        break
                    x += 1
                row.append(x)
                local_used.update(new_diffs)
            rows.append(row)
            used.update(local_used)
        best_rows = rows
        current_rows = [r[:] for r in rows]
        current_used = set(used)
        best_max = get_max(rows)

    if best_max == lb:
        return best_rows

    # ---- 2. Row-rebuild local search with difference-pressure selection ----
    rng = random.Random(42)
    current_max = best_max

    def rebuild_row(k, forbidden_diffs, rng, max_val_limit, bias=0.0):
        """
        Builds a row of size k+1 starting at 0, strictly increasing,
        avoiding forbidden_diffs, with max entry <= max_val_limit.
        Uses difference-pressure candidate selection.
        """
        row = [0]
        local_used = set()

        for pos in range(k):
            last = row[-1]
            remaining = k - pos - 1
            max_x_allowed = max_val_limit - remaining

            if last + 1 > max_x_allowed:
                return None

            candidates = []
            x = last + 1
            while x <= max_x_allowed:
                new_diffs = {x - a for a in row}
                if not (new_diffs & local_used) and not (new_diffs & forbidden_diffs):
                    max_new_diff = max(new_diffs)
                    score = x + 0.3 * max_new_diff
                    candidates.append((score, x, new_diffs))
                x += 1
                if len(candidates) >= 20:
                    break

            if not candidates:
                return None

            candidates.sort(key=lambda c: c[0])
            top_n = min(10, len(candidates))

            if bias > 0 and rng.random() < bias:
                idx = rng.randint(0, top_n - 1)
            else:
                idx = 0

            _, x, new_diffs = candidates[idx]
            row.append(x)
            local_used.update(new_diffs)

        return row

    def rebuild_multi_row(indices, forbidden_diffs, rng, max_val_limit):
        """
        Rebuild multiple rows together. Remove their diffs, rebuild each,
        checking against the combined forbidden set + already-rebuilt rows.
        """
        old_rows = [current_rows[i] for i in indices]
        old_diffs = set()
        for r in old_rows:
            old_diffs.update(get_diffs(r))

        # Remove old diffs from current_used
        current_used.difference_update(old_diffs)

        new_rows = []
        for idx in indices:
            row = rebuild_row(k, current_used, rng, max_val_limit, bias=0.5)
            if row is None:
                # Restore
                current_used.update(old_diffs)
                return None
            new_rows.append(row)
            current_used.update(get_diffs(row))

        # Replace in current_rows
        for i, idx in enumerate(indices):
            current_rows[idx] = new_rows[i]

        return new_rows

    no_improve_count = 0
    max_no_improve = 200

    while not time_up() and no_improve_count < max_no_improve:
        # Decide: single-row or multi-row rebuild
        if n_rows >= 2 and rng.random() < 0.3:
            # Multi-row: pick 2 rows
            i = rng.randint(0, n_rows - 1)
            j = rng.randint(0, n_rows - 1)
            while j == i:
                j = rng.randint(0, n_rows - 1)
            indices = [i, j]
        else:
            i = rng.randint(0, n_rows - 1)
            indices = [i]

        # Determine target max
        max_in_selected = max(max(current_rows[idx]) for idx in indices)
        if max_in_selected == current_max:
            target_max = current_max - 1
        else:
            target_max = current_max

        if target_max < k:
            target_max = k

        if len(indices) == 1:
            idx = indices[0]
            old_row = current_rows[idx]
            old_diffs = get_diffs(old_row)
            current_used.difference_update(old_diffs)

            new_row = rebuild_row(k, current_used, rng, target_max, bias=0.5)

            if new_row is None:
                # Try reshuffle at same max
                new_row = rebuild_row(k, current_used, rng, current_max, bias=0.8)

            if new_row is None:
                current_used.update(old_diffs)
                no_improve_count += 1
                continue

            new_diffs = get_diffs(new_row)
            new_row_max = max(new_row)
            # Recompute global max
            other_max = max(max(current_rows[r]) for r in range(n_rows) if r != idx)
            new_global_max = max(new_row_max, other_max)

            if new_global_max < current_max:
                current_rows[idx] = new_row
                current_used.update(new_diffs)
                current_max = new_global_max
                best_max = new_global_max
                best_rows = [r[:] for r in current_rows]
                no_improve_count = 0
            elif new_global_max == current_max:
                # Accept reshuffle with some probability to escape local minima
                if rng.random() < 0.5:
                    current_rows[idx] = new_row
                    current_used.update(new_diffs)
                    no_improve_count += 1
                else:
                    current_used.update(old_diffs)
                    no_improve_count += 1
            else:
                current_used.update(old_diffs)
                no_improve_count += 1
        else:
            # Multi-row rebuild
            result = rebuild_multi_row(indices, current_used, rng, target_max)
            if result is None:
                # Try at current_max
                result = rebuild_multi_row(indices, current_used, rng, current_max)

            if result is None:
                no_improve_count += 1
                continue

            new_global_max = get_max(current_rows)
            if new_global_max < current_max:
                current_max = new_global_max
                best_max = new_global_max
                best_rows = [r[:] for r in current_rows]
                no_improve_count = 0
            else:
                no_improve_count += 1

    # ---- 3. Element-level SA polish in the final phase ----
    if not time_up() and best_max > lb:
        rng2 = random.Random(99)
        current_rows2 = [r[:] for r in best_rows]
        current_used2 = set()
        for row in current_rows2:
            current_used2.update(get_diffs(row))
        current_max2 = best_max

        T = 5.0
        T_min = 0.1
        cooling = 0.9998
        polish_deadline = start_time + 116.0

        def try_element_move():
            nonlocal current_rows2, current_used2, current_max2, T

            r_idx = rng2.randint(0, n_rows - 1)
            row = current_rows2[r_idx]
            pos = rng2.randint(1, len(row) - 1)

            old_val = row[pos]
            min_val = row[pos - 1] + 1
            max_val = row[pos + 1] - 1 if pos < len(row) - 1 else current_max2

            if min_val > max_val:
                return False

            for _ in range(15):
                delta = rng2.randint(-4, 4)
                new_val = old_val + delta

                if new_val < min_val or new_val > max_val:
                    continue

                # Old diffs involving this element
                old_diffs = set()
                for i in range(len(row)):
                    if i != pos:
                        d = abs(row[pos] - row[i])
                        old_diffs.add(d)

                current_used2.difference_update(old_diffs)

                valid = True
                new_diffs = set()
                for i in range(len(row)):
                    if i != pos:
                        d = abs(new_val - row[i])
                        if d in current_used2:
                            valid = False
                            break
                        new_diffs.add(d)

                if not valid:
                    current_used2.update(old_diffs)
                    continue

                if len(new_diffs) != len(new_diffs):
                    current_used2.update(old_diffs)
                    continue

                # New max
                new_max = 0
                for i, v in enumerate(row):
                    val = new_val if i == pos else v
                    if val > new_max:
                        new_max = val
                if new_val > new_max:
                    new_max = new_val

                if new_max <= current_max2:
                    row[pos] = new_val
                    current_used2.update(new_diffs)
                    if new_max < current_max2:
                        current_max2 = new_max
                        best_max = new_max
                        best_rows = [r[:] for r in current_rows2]
                    return True
                else:
                    prob = math.exp(-(new_max - current_max2) / T)
                    if rng2.random() < prob:
                        row[pos] = new_val
                        current_used2.update(new_diffs)
                        current_max2 = new_max
                        if current_max2 > best_max:
                            row[pos] = old_val
                            current_used2.difference_update(new_diffs)
                            current_used2.update(old_diffs)
                            current_max2 = best_max
                        return True
                    else:
                        current_used2.update(old_diffs)
                        continue

            return False

        while time.time() < polish_deadline and T > T_min:
            try_element_move()
            T *= cooling

    # Final validation
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
        return True

    if not validate(best_rows, n_rows, k):
        res = greedy_construct(0)
        if res:
            best_rows, _ = res

    return best_rows

# EVOLVE-BLOCK-END
