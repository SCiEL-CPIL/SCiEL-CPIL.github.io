"""Construct difference triangle sets (n_rows, k) minimizing the largest entry."""
import time
import random
import math

_DEADLINE = None
_TIME_LIMIT = 115.0


def _time_left():
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
    # Maximum possible difference we care about
    # For scope m, differences are in [1, m]
    # We'll use a bytearray for O(1) checks

    def make_bitset(scope):
        return bytearray(scope + 2)  # 1-indexed, index 0 unused

    def check_and_mark(used_bits, scope, row, new_val):
        """Check if adding new_val to row creates conflicts. Returns True if OK."""
        for a in row:
            d = new_val - a
            if d > scope:
                return False
            if used_bits[d]:
                return False
        return True

    def mark_diffs(used_bits, row, new_val):
        """Mark differences created by adding new_val to row."""
        for a in row:
            d = new_val - a
            used_bits[d] = 1

    def unmark_diffs(used_bits, row, new_val):
        """Unmark differences for new_val in row."""
        for a in row:
            d = new_val - a
            if d <= len(used_bits) - 1:
                used_bits[d] = 0

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
        """
        Try to construct all rows with max entry <= target_scope using bitset.
        Returns (rows, success).
        """
        used_bits = bytearray(target_scope + 2)
        rows = []
        for row_idx in range(n_rows):
            if _time_left() < 3.0:
                return None, False
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Find valid candidates within [x+1, target_scope]
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

                # Choose with bias
                if rng.random() < bias_smallest:
                    x = valid[0]
                else:
                    # Weighted toward smaller
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

    # --- Local search with bitset: try to reduce each element ---
    def local_search_bitset(rows, target_scope, max_iters=300):
        """
        Local search: try to reduce elements to smaller valid values.
        Uses bitset for fast conflict checking.
        Returns (rows, improved).
        """
        rows = [list(r) for r in rows]
        scope = target_scope
        used_bits = bytearray(scope + 2)

        # Build initial used_bits
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
            # Process rows in order of largest entry first
            row_order = sorted(range(len(rows)), key=lambda i: rows[i][-1], reverse=True)

            for ri in row_order:
                if _time_left() < 2.0:
                    break
                row = rows[ri]

                # Try to reduce each element from right to left
                for pos in range(len(row) - 1, 0, -1):
                    current = row[pos]
                    # Find the smallest valid value for this position
                    # Must be > row[pos-1] and < row[pos+1] if pos < len(row)-1
                    lower = row[pos - 1] + 1
                    upper = current  # try to reduce to current or below

                    # Try values from lower up to current
                    for new_val in range(lower, current):
                        # Check if new_val is valid
                        # Remove current's differences
                        for j in range(len(row)):
                            if j == pos:
                                continue
                            d = abs(row[j] - current)
                            if d > 0 and d <= scope:
                                used_bits[d] = 0

                        # Check new_val
                        ok = True
                        for j in range(len(row)):
                            if j == pos:
                                continue
                            d = abs(row[j] - new_val)
                            if d > scope or used_bits[d]:
                                ok = False
                                break

                        if ok:
                            # Also check ordering
                            if pos > 0 and new_val <= row[pos - 1]:
                                ok = False
                            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                                ok = False

                        if ok:
                            # Mark new_val's differences
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
                            # Restore current's differences
                            for j in range(len(row)):
                                if j == pos:
                                    continue
                                d = abs(row[j] - current)
                                if d > 0 and d <= scope:
                                    used_bits[d] = 1

            if not any_progress:
                break

        return rows, improved

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
    # Phase 1: Scope-budgeted construction with decreasing targets
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
                    # Apply local search
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

        # Randomized greedy with bitset
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
                    # Try larger
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

    # Phase 3: Intensive local search on best
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
