import time
import random
import math

_TIME_LIMIT = 140.0  # safety margin well under the 150s kill
_START_TIME = time.time()

def _time_left():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _validate(rows):
    """Check that all positive differences within each row are distinct across all rows."""
    used = 0
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if used & (1 << d):
                    return False
                used |= (1 << d)
    return True

def _scope(rows):
    return max(max(r) for r in rows) if rows else 0

def _row_diff_mask(row):
    """Compute bitmask of all positive differences in a row."""
    mask = 0
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            mask |= (1 << d)
    return mask

def _all_diffs_except(rows, skip_idx):
    """Bitmask of all positive differences in all rows except rows[skip_idx]."""
    mask = 0
    for i in range(len(rows)):
        if i == skip_idx:
            continue
        row = rows[i]
        n = len(row)
        for a in range(n):
            for b in range(a + 1, n):
                mask |= (1 << (row[b] - row[a]))
    return mask

def _greedy_row(used_mask, k, rng, strategy="shuffled"):
    """
    Build one row greedily.
    used_mask: bitmask of differences used by other rows.
    strategy: 'pure', 'shuffled', 'lookahead'
    """
    row = [0]
    x = 0
    local_mask = 0

    while len(row) < k + 1:
        if strategy == "pure":
            x += 1
            while True:
                new_diffs = 0
                valid = True
                for a in row:
                    d = x - a
                    if local_mask & (1 << d) or used_mask & (1 << d):
                        valid = False
                        break
                    new_diffs |= (1 << d)
                if valid:
                    break
                x += 1
            row.append(x)
            local_mask |= new_diffs
        elif strategy == "shuffled":
            base = x + 1
            window = 80
            candidates = list(range(base, base + window))
            if rng:
                rng.shuffle(candidates)
            found = False
            for cand in candidates:
                new_diffs = 0
                valid = True
                for a in row:
                    d = cand - a
                    if local_mask & (1 << d) or used_mask & (1 << d):
                        valid = False
                        break
                    new_diffs |= (1 << d)
                if valid:
                    row.append(cand)
                    local_mask |= new_diffs
                    x = cand
                    found = True
                    break
            if not found:
                cand = base + window
                while True:
                    new_diffs = 0
                    valid = True
                    for a in row:
                        d = cand - a
                        if local_mask & (1 << d) or used_mask & (1 << d):
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    if valid:
                        row.append(cand)
                        local_mask |= new_diffs
                        x = cand
                        break
                    cand += 1
        elif strategy == "lookahead":
            base = x + 1
            valid_cands = []
            for cand in range(base, base + 60):
                new_diffs = 0
                valid = True
                for a in row:
                    d = cand - a
                    if local_mask & (1 << d) or used_mask & (1 << d):
                        valid = False
                        break
                    new_diffs |= (1 << d)
                if valid:
                    valid_cands.append((cand, new_diffs))
                if len(valid_cands) >= 15:
                    break

            if not valid_cands:
                cand = base
                while True:
                    new_diffs = 0
                    valid = True
                    for a in row:
                        d = cand - a
                        if local_mask & (1 << d) or used_mask & (1 << d):
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    if valid:
                        valid_cands.append((cand, new_diffs))
                        break
                    cand += 1

            # Score: prefer smaller cand, with a light future-options tiebreak
            best_cand, best_mask = valid_cands[0]
            best_score = -1
            for cand, cmask in valid_cands[:6]:
                # count a couple of valid next options for tiebreak
                next_count = 0
                for nc in range(cand + 1, cand + 20):
                    ok = True
                    for a in row:
                        d = nc - a
                        if local_mask & (1 << d) or used_mask & (1 << d) or cmask & (1 << d):
                            ok = False
                            break
                    d2 = nc - cand
                    if cmask & (1 << d2):
                        ok = False
                    if ok:
                        next_count += 1
                    if next_count >= 4:
                        break
                score = next_count * 100 - cand
                if score > best_score:
                    best_score = score
                    best_cand = cand
                    best_mask = cmask
            row.append(best_cand)
            local_mask |= best_mask
            x = best_cand

    return row, local_mask

def _construct_set(n_rows, k, rng, strategy):
    """Construct a full set of rows."""
    used_mask = 0
    rows = []
    for _ in range(n_rows):
        row, mask = _greedy_row(used_mask, k, rng, strategy)
        rows.append(row)
        used_mask |= mask
    return rows

def _backtrack_row(used_mask, k, max_scope, rng, time_budget=0.35):
    """
    Build one row with all entries <= max_scope using DFS + bitmask pruning.
    used_mask: bitmask of differences used by other rows (fixed).
    Returns row or None.
    """
    row = [0]
    local_mask = 0
    t0 = time.time()
    nodes = [0]
    limit = 300000

    def bt():
        if _time_left() < 1.0:
            return False
        if (time.time() - t0) > time_budget:
            return False
        nodes[0] += 1
        if nodes[0] > limit:
            return False
        if len(row) == k + 1:
            return True
        start = row[-1] + 1
        # dynamic lower-bound-ish: prune candidates that can't fit remaining entries
        remaining = (k + 1) - len(row)
        # need at least (remaining-1) more steps of at least +1, so cand <= max_scope - (remaining-1)
        hi = min(max_scope, max_scope - (remaining - 1))
        if hi < start:
            return False
        cands = list(range(start, hi + 1))
        if len(cands) > 24 and rng is not None:
            rng.shuffle(cands[12:])
        for x in cands:
            new_diffs = 0
            valid = True
            for a in row:
                d = x - a
                if local_mask & (1 << d) or used_mask & (1 << d):
                    valid = False
                    break
                new_diffs |= (1 << d)
            if not valid:
                continue
            row.append(x)
            nm = local_mask | new_diffs
            if bt():
                return True
            row.pop()
        return False

    if bt():
        return row
    return None

def _local_search_reduce(rows, n_rows, k, rng, max_iters=50000, rebuild=True):
    """
    Local search aimed at reducing the maximum element.
    If rebuild=True, periodically attempt a full backtracking rebuild of the max row
    at a tightened scope cap (CSP-style constraint propagation).
    """
    rows = [r[:] for r in rows]
    best_scope = _scope(rows)
    row_masks = [_row_diff_mask(r) for r in rows]

    for iters in range(max_iters):
        if _time_left() < 2.0:
            break

        # Find row with max element
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        target = best_scope - 1
        if target < k + 1:
            break

        other_mask = 0
        for i in range(n_rows):
            if i != max_idx:
                other_mask |= row_masks[i]

        row = rows[max_idx]
        improved = False

        # (A) Rebuild the max row at a tightened cap (can drop scope by >1)
        if rebuild and (iters % 3 == 0):
            # try caps from target down to a floor
            floor = max(k + 1, row[-2] + 1)
            for cap in range(target, floor - 1, -1):
                if cap < floor:
                    break
                new_row = _backtrack_row(other_mask, k, cap, rng, time_budget=0.30)
                if new_row is not None:
                    rows[max_idx] = new_row
                    row_masks[max_idx] = _row_diff_mask(new_row)
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    improved = True
                    break
            if improved:
                continue

        # (B) Try to reduce the last element (greedy downward scan)
        start_val = min(current_max - 1, target)
        end_val = row[-2] + 1
        for new_val in range(start_val, end_val - 1, -1):
            if new_val <= row[-2]:
                break
            new_row = row[:-1] + [new_val]
            new_mask = 0
            valid = True
            n = len(new_row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = new_row[j] - new_row[i]
                    if new_mask & (1 << d):
                        valid = False
                        break
                    new_mask |= (1 << d)
            if not valid:
                continue
            if new_mask & other_mask:
                continue
            rows[max_idx] = new_row
            row_masks[max_idx] = new_mask
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
            improved = True
            break

        if not improved:
            # (C) Perturb an interior element to free up space
            if len(row) > 2:
                pos = rng.randint(1, len(row) - 2)
                old_val = row[pos]
                lower = row[pos - 1] + 1
                upper = row[pos + 1] - 1
                tried = 0
                while tried < 12:
                    new_val = rng.randint(lower, upper)
                    if new_val == old_val:
                        tried += 1
                        continue
                    new_row = row[:pos] + [new_val] + row[pos + 1:]
                    new_mask = 0
                    valid = True
                    n = len(new_row)
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = new_row[j] - new_row[i]
                            if new_mask & (1 << d):
                                valid = False
                                break
                            new_mask |= (1 << d)
                    if not valid:
                        tried += 1
                        continue
                    if new_mask & other_mask:
                        tried += 1
                        continue
                    new_scope_row = max(new_row)
                    if new_scope_row <= current_max:
                        rows[max_idx] = new_row
                        row_masks[max_idx] = new_mask
                        improved = True
                        break
                    tried += 1

    return rows, best_scope

def _simulated_annealing(rows, n_rows, k, rng, max_iters=20000):
    """Simulated annealing to find better configurations."""
    rows = [r[:] for r in rows]
    best_rows = [r[:] for r in rows]
    best_scope = _scope(rows)
    current_scope = best_scope

    row_masks = [_row_diff_mask(r) for r in rows]

    start_temp = 50.0
    end_temp = 0.1
    cooling = 0.999
    temp = start_temp

    for iters in range(max_iters):
        if _time_left() < 2.0:
            break

        r_idx = rng.randint(0, n_rows - 1)
        row = rows[r_idx]
        if len(row) <= 1:
            continue
        pos = rng.randint(1, len(row) - 1)
        current_val = row[pos]

        lower = row[pos - 1] + 1
        upper = row[pos + 1] - 1 if pos < len(row) - 1 else current_scope + 10
        if upper < lower:
            continue

        new_val = rng.randint(lower, upper)
        if new_val == current_val:
            continue

        new_row = row[:pos] + [new_val] + row[pos + 1:]
        if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
            continue

        new_mask = 0
        valid = True
        n = len(new_row)
        for i in range(n):
            for j in range(i + 1, n):
                d = new_row[j] - new_row[i]
                if new_mask & (1 << d):
                    valid = False
                    break
                new_mask |= (1 << d)
        if not valid:
            continue

        other_mask = 0
        for i in range(n_rows):
            if i != r_idx:
                other_mask |= row_masks[i]
        if new_mask & other_mask:
            continue

        new_scope = max(max(new_row), max(max(rows[i]) for i in range(n_rows) if i != r_idx))

        if new_scope < current_scope:
            rows[r_idx] = new_row
            row_masks[r_idx] = new_mask
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
        else:
            delta = new_scope - current_scope
            if temp > 0 and rng.random() < math.exp(-delta / temp):
                rows[r_idx] = new_row
                row_masks[r_idx] = new_mask
                current_scope = new_scope

        temp = max(end_temp, temp * cooling)

    return best_rows, best_scope

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()

    rng = random.Random(42)

    best_rows = None
    best_scope = float('inf')

    strategies = ["pure", "shuffled", "lookahead"]

    # Phase 1: Diverse constructions
    for strategy in strategies:
        if _time_left() < 12.0:
            break
        rows = _construct_set(n_rows, k, rng, strategy)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]

    # More randomized restarts
    attempts = 0
    while _time_left() > 40.0 and attempts < 60:
        attempts += 1
        strategy = rng.choice(strategies)
        rows = _construct_set(n_rows, k, rng, strategy)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
            if best_scope <= 112:
                break

    # Phase 2: Local Search with rebuild to reduce scope
    if _time_left() > 30.0 and best_rows:
        rows_ls, scope_ls = _local_search_reduce(best_rows, n_rows, k, rng, max_iters=20000, rebuild=True)
        if scope_ls < best_scope:
            best_scope = scope_ls
            best_rows = rows_ls

    # Phase 3: Simulated Annealing polish
    if _time_left() > 25.0 and best_rows:
        rows_sa, scope_sa = _simulated_annealing(best_rows, n_rows, k, rng, max_iters=30000)
        if scope_sa < best_scope:
            best_scope = scope_sa
            best_rows = rows_sa

    # Phase 4: High-precision polish (tighter, more iterations)
    if _time_left() > 15.0 and best_rows:
        rows_ls2, scope_ls2 = _local_search_reduce(best_rows, n_rows, k, rng, max_iters=30000, rebuild=True)
        if scope_ls2 < best_scope:
            best_scope = scope_ls2
            best_rows = rows_ls2
        if _time_left() > 8.0:
            rows_sa2, scope_sa2 = _simulated_annealing(best_rows, n_rows, k, rng, max_iters=10000)
            if scope_sa2 < best_scope:
                best_scope = scope_sa2
                best_rows = rows_sa2

    # Final validation
    if best_rows is None or not _validate(best_rows):
        used_mask = 0
        rows = []
        for _ in range(n_rows):
            row, mask = _greedy_row(used_mask, k, None, "pure")
            rows.append(row)
            used_mask |= mask
        return rows

    return best_rows
