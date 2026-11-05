"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math
import sys

_TIME_LIMIT = 115.0
_DEADLINE = None


def _time_left():
    global _DEADLINE
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - time.time())


def construct(n_rows, k):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

    total_diffs = n_rows * k * (k + 1) // 2

    # Counting lower bound
    lower_bound = 0
    s = 0
    while s < total_diffs:
        lower_bound += 1
        s += lower_bound

    rng = random.Random(42)

    # --- Bitset-based used-difference tracking ---
    def make_bitset(scope):
        return bytearray(scope + 2)

    # --- Deterministic greedy baseline ---
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    best_rows = greedy_construct()
    best_max = max(max(r) for r in best_rows)

    # --- Scope-budgeted construction with bitset ---
    def scope_construct_bitset(target_scope, rng, bias_smallest=0.6):
        used_bits = bytearray(target_scope + 2)
        rows = []
        for row_idx in range(n_rows):
            if _time_left() < 3.0:
                return None, False
            row = [0]
            x = 0
            while len(row) < k + 1:
                valid = []
                for candidate in range(x + 1, target_scope + 1):
                    ok = True
                    for a in row:
                        d = candidate - a
                        if d > target_scope or used_bits[d]:
                            ok = False
                            break
                    if ok:
                        valid.append(candidate)
                        if len(valid) >= 20:
                            break

                if not valid:
                    return None, False

                if rng.random() < bias_smallest:
                    x = valid[0]
                else:
                    w = [1.0 / (i + 1) for i in range(len(valid))]
                    total_w = sum(w)
                    r = rng.random() * total_w
                    cum = 0.0
                    x = valid[-1]
                    for i in range(len(valid)):
                        cum += w[i]
                        if r <= cum:
                            x = valid[i]
                            break

                row.append(x)
                for a in row[:-1]:
                    used_bits[x - a] = 1
            rows.append(row)

        return rows, True

    # --- Local search: try to reduce each element ---
    def local_search_bitset(rows, target_scope, max_iters=300):
        rows = [list(r) for r in rows]
        scope = target_scope
        used_bits = bytearray(scope + 2)

        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= scope:
                        used_bits[d] = 1

        improved = False
        for _ in range(max_iters):
            if _time_left() < 2.0:
                break

            any_progress = False
            row_order = sorted(range(len(rows)), key=lambda i: rows[i][-1], reverse=True)

            for ri in row_order:
                if _time_left() < 2.0:
                    break
                row = rows[ri]

                for pos in range(len(row) - 1, 0, -1):
                    current = row[pos]
                    lower = row[pos - 1] + 1

                    for new_val in range(lower, current):
                        # Remove current's differences
                        for j in range(len(row)):
                            if j == pos:
                                continue
                            d = abs(row[j] - current)
                            if d > 0 and d <= scope:
                                used_bits[d] = 0

                        ok = True
                        for j in range(len(row)):
                            if j == pos:
                                continue
                            d = abs(row[j] - new_val)
                            if d > scope or used_bits[d]:
                                ok = False
                                break

                        if ok:
                            if pos > 0 and new_val <= row[pos - 1]:
                                ok = False
                            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                                ok = False

                        if ok:
                            for j in range(len(row)):
                                if j == pos:
                                    continue
                                d = abs(row[j] - new_val)
                                if d > 0 and d <= scope:
                                    used_bits[d] = 1
                            row[pos] = new_val
                            improved = True
                            any_progress = True
                            break
                        else:
                            for j in range(len(row)):
                                if j == pos:
                                    continue
                                d = abs(row[j] - current)
                                if d > 0 and d <= scope:
                                    used_bits[d] = 1

            if not any_progress:
                break

        return rows, improved

    # --- Backtracking row construction (from inspiration) ---
    def backtrack_row(used_bits, k, max_scope, rng, depth=0):
        """
        Try to build one row with all entries <= max_scope using backtracking.
        Returns row or None.
        """
        if _time_left() < 2.0:
            return None

        row = [0]

        def bt():
            nonlocal row
            if len(row) == k + 1:
                return True
            if _time_left() < 1.0:
                return False

            start = row[-1] + 1
            # Limit candidates to a window for efficiency
            window_end = min(max_scope + 1, start + 80)
            candidates = list(range(start, window_end))

            # Shuffle for diversity
            if len(candidates) > 5:
                rng.shuffle(candidates[3:])

            for x in candidates:
                if x > max_scope:
                    break
                # Check conflicts
                ok = True
                for a in row:
                    d = x - a
                    if d > max_scope or used_bits[d]:
                        ok = False
                        break
                if not ok:
                    continue

                # Mark
                row.append(x)
                for a in row[:-1]:
                    used_bits[x - a] = 1

                if bt():
                    return True

                # Unmark
                for a in row[:-1]:
                    used_bits[x - a] = 0
                row.pop()

            return False

        if bt():
            return row
        return None

    # --- Rebuild a row given other rows' differences ---
    def rebuild_row(used_other_bits, k, max_scope, rng):
        """Rebuild a single row given the differences used by other rows."""
        return backtrack_row(used_other_bits, k, max_scope, rng)

    # --- Compute used bits for all rows except one ---
    def all_diffs_except(rows, skip_idx, scope):
        used = bytearray(scope + 2)
        for idx, row in enumerate(rows):
            if idx == skip_idx:
                continue
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d <= scope:
                        used[d] = 1
        return used

    # --- Simulated Annealing for row rebuilding (new structural idea) ---
    def sa_rebuild_row(used_other_bits, k, max_scope, rng, iterations=5000):
        """
        Use simulated annealing to find a valid row with entries <= max_scope.
        State: a list of k+1 increasing integers starting at 0.
        Energy: number of conflicts (differences that are in used_other_bits or duplicated within row).
        """
        if _time_left() < 2.0:
            return None

        # Initial solution: greedy
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            ok = True
            for a in row:
                d = x - a
                if d > max_scope or used_other_bits[d]:
                    ok = False
                    break
            if ok:
                row.append(x)
            if len(row) >= k + 1:
                break
        if len(row) < k + 1:
            # Pad with large values
            while len(row) < k + 1:
                x += 1
                row.append(x)
            return None  # Greedy failed, SA may not help much

        def energy(r):
            """Count conflicts: differences in used_other_bits or duplicated within row."""
            diffs = set()
            count = 0
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d > max_scope:
                        count += 1
                        continue
                    if used_other_bits[d]:
                        count += 1
                    if d in diffs:
                        count += 1
                    diffs.add(d)
            return count

        current_energy = energy(row)
        if current_energy == 0:
            return row

        best_row = row[:]
        best_energy = current_energy

        T = 10.0
        cooling = 0.999

        for _ in range(iterations):
            if _time_left() < 1.0:
                break

            # Pick a random position to modify (not the first)
            pos = rng.randint(1, len(row) - 1)
            old_val = row[pos]

            # Generate new value: nearby or random
            if rng.random() < 0.7:
                # Local move
                delta = rng.randint(-5, 5)
                new_val = old_val + delta
            else:
                # Larger jump
                new_val = rng.randint(row[pos - 1] + 1, max_scope)

            # Enforce ordering
            if new_val <= row[pos - 1]:
                new_val = row[pos - 1] + 1
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                new_val = row[pos + 1] - 1
            if new_val <= row[pos - 1] or (pos < len(row) - 1 and new_val >= row[pos + 1]):
                continue

            row[pos] = new_val
            new_energy = energy(row)

            delta_e = new_energy - current_energy
            if delta_e <= 0 or rng.random() < math.exp(-delta_e / max(T, 0.001)):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_row = row[:]
                    if best_energy == 0:
                        return best_row
            else:
                row[pos] = old_val

            T *= cooling

        return best_row if best_energy == 0 else None

    # --- Validate ---
    def validate(rows):
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
                    if row[j] <= row[i]:
                        return False
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    # --- Main search loop ---
    # Phase 1: Scope-budgeted construction
    for scope_attempt in range(40):
        if _time_left() < 10.0:
            break

        for frac in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75]:
            target = int(best_max * frac)
            if target < lower_bound:
                continue

            for trial in range(8):
                if _time_left() < 5.0:
                    break
                rows, success = scope_construct_bitset(target, rng, bias_smallest=0.5 + 0.05 * trial)
                if not success:
                    continue
                cur_max = max(max(r) for r in rows)
                if cur_max < best_max:
                    if _time_left() > 5.0:
                        improved_rows, improved = local_search_bitset(rows, cur_max, max_iters=200)
                        if improved:
                            new_max = max(max(r) for r in improved_rows)
                            if new_max < best_max:
                                best_max = new_max
                                best_rows = improved_rows
                        else:
                            best_max = cur_max
                            best_rows = rows
                    else:
                        best_max = cur_max
                        best_rows = rows
                    break

    # Phase 2: Randomized restarts with bitset
    for attempt in range(100):
        if _time_left() < 8.0:
            break

        used_bits = bytearray(best_max + 2)
        rows = []
        success = True
        for row_idx in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                window = max(40, 80 - 8 * row_idx)
                valid = []
                for candidate in range(x + 1, min(x + window + 1, best_max + 1)):
                    ok = True
                    for a in row:
                        d = candidate - a
                        if d > best_max or used_bits[d]:
                            ok = False
                            break
                    if ok:
                        valid.append(candidate)
                        if len(valid) >= 15:
                            break

                if not valid:
                    candidate = x + 1
                    while candidate <= best_max:
                        ok = True
                        for a in row:
                            d = candidate - a
                            if d > best_max or used_bits[d]:
                                ok = False
                                break
                        if ok:
                            valid.append(candidate)
                            break
                        candidate += 1

                if not valid:
                    success = False
                    break

                if len(valid) == 1:
                    x = valid[0]
                else:
                    if rng.random() < 0.5:
                        x = valid[0]
                    else:
                        w = [1.0 / (i + 1) for i in range(len(valid))]
                        total_w = sum(w)
                        r = rng.random() * total_w
                        cum = 0.0
                        x = valid[-1]
                        for i in range(len(valid)):
                            cum += w[i]
                            if r <= cum:
                                x = valid[i]
                                break

                row.append(x)
                for a in row[:-1]:
                    d = x - a
                    if d <= best_max:
                        used_bits[d] = 1
            rows.append(row)

        if success:
            cur_max = max(max(r) for r in rows)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = rows
                if _time_left() > 5.0:
                    improved_rows, improved = local_search_bitset(best_rows, best_max, max_iters=150)
                    if improved:
                        new_max = max(max(r) for r in improved_rows)
                        if new_max < best_max:
                            best_max = new_max
                            best_rows = improved_rows

    # Phase 3: Row-rebuilding with backtracking and SA (new structural approach)
    if _time_left() > 15.0:
        rows_copy = [list(r) for r in best_rows]
        scope = best_max

        for rebuild_round in range(20):
            if _time_left() < 10.0:
                break

            # Find the row with the largest entry
            max_idx = max(range(n_rows), key=lambda i: rows_copy[i][-1])
            current_max_entry = rows_copy[max_idx][-1]

            # Try to reduce the scope
            target_scope = current_max_entry - 1
            if target_scope < k + 1:
                break

            # Compute used bits for other rows
            used_other = all_diffs_except(rows_copy, max_idx, target_scope)

            # Try backtracking first
            new_row = backtrack_row(used_other, k, target_scope, rng)
            if new_row is not None:
                rows_copy[max_idx] = new_row
                new_scope = max(max(r) for r in rows_copy)
                if new_scope < best_max:
                    best_max = new_scope
                    best_rows = [list(r) for r in rows_copy]
                    continue

            # Try SA
            if _time_left() > 5.0:
                sa_row = sa_rebuild_row(used_other, k, target_scope, rng, iterations=3000)
                if sa_row is not None:
                    rows_copy[max_idx] = sa_row
                    new_scope = max(max(r) for r in rows_copy)
                    if new_scope < best_max:
                        best_max = new_scope
                        best_rows = [list(r) for r in rows_copy]
                        continue

            # Try reducing individual elements of the max row
            row = rows_copy[max_idx]
            improved = False
            for pos in range(len(row) - 1, 0, -1):
                current = row[pos]
                lower = row[pos - 1] + 1
                for new_val in range(lower, current):
                    # Check if new_val is valid
                    ok = True
                    for j in range(len(row)):
                        if j == pos:
                            continue
                        d = abs(row[j] - new_val)
                        if d > target_scope or used_other[d]:
                            ok = False
                            break
                    if ok:
                        if pos < len(row) - 1 and new_val >= row[pos + 1]:
                            ok = False
                    if ok:
                        row[pos] = new_val
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                # Try next row
                continue

            new_scope = max(max(r) for r in rows_copy)
            if new_scope < best_max:
                best_max = new_scope
                best_rows = [list(r) for r in rows_copy]

    # Phase 4: Intensive local search on best
    if _time_left() > 5.0:
        improved_rows, improved = local_search_bitset(best_rows, best_max, max_iters=400)
        if improved:
            new_max = max(max(r) for r in improved_rows)
            if new_max < best_max:
                best_max = new_max
                best_rows = improved_rows

    if _time_left() > 3.0:
        improved_rows, improved = local_search_bitset(best_rows, best_max, max_iters=200)
        if improved:
            new_max = max(max(r) for r in improved_rows)
            if new_max < best_max:
                best_max = new_max
                best_rows = improved_rows

    # Final validation
    if not validate(best_rows):
        best_rows = greedy_construct()

    return best_rows
