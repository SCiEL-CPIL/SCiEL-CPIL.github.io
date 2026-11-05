"""Construct difference triangle sets (n_rows, k) minimizing the largest entry.

Structural core (from the high-scoring inspiration): row-rebuild moves.
We maintain a full set of rows; a move removes one row and rebuilds it with a
randomized greedy constrained to a target maximum, accepting the move only if
it reduces the global scope. A binary-search tightening loop drives the scope
down. Fast polish (from the current program) then reduces individual entries.
"""
import time
import random
import math

_TIME_LIMIT = 120.0
_DEADLINE = None
_DEADLINE_SET = False


def _time_left():
    global _DEADLINE, _DEADLINE_SET
    if not _DEADLINE_SET:
        _DEADLINE = time.time() + _TIME_LIMIT
        _DEADLINE_SET = True
    return max(0.0, _DEADLINE - time.time())


def _validate(rows):
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    return max(max(r) for r in rows) if rows else 0


def _row_diffs(row):
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s


def _all_diffs_except(rows, skip_idx):
    s = set()
    for idx, row in enumerate(rows):
        if idx == skip_idx:
            continue
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                s.add(row[j] - row[i])
    return s


def _greedy_row(used, k, rng=None, strategy="pure"):
    row = [0]
    x = 0
    while len(row) < k + 1:
        if strategy == "pure":
            x += 1
            while True:
                new = {x - a for a in row}
                if not new & used:
                    break
                x += 1
            row.append(x)
            used |= {x - a for a in row[:-1]}
        elif strategy == "shuffled":
            base = x + 1
            window = 40
            candidates = list(range(base, base + window))
            if rng:
                rng.shuffle(candidates)
            found = False
            for cand in candidates:
                new = {cand - a for a in row}
                if not new & used:
                    row.append(cand)
                    used |= {cand - a for a in row[:-1]}
                    x = cand
                    found = True
                    break
            if not found:
                cand = base + window
                while True:
                    new = {cand - a for a in row}
                    if not new & used:
                        row.append(cand)
                        used |= {cand - a for a in row[:-1]}
                        x = cand
                        break
                    cand += 1
        elif strategy == "lookahead":
            base = x + 1
            valid_cands = []
            for cand in range(base, base + 50):
                new = {cand - a for a in row}
                if not new & used:
                    valid_cands.append(cand)
                if len(valid_cands) >= 8:
                    break
            if not valid_cands:
                cand = base
                while True:
                    new = {cand - a for a in row}
                    if not new & used:
                        valid_cands.append(cand)
                        break
                    cand += 1
            best_cand = valid_cands[0]
            best_score = -1
            for cand in valid_cands[:6]:
                new = {cand - a for a in row}
                next_count = 0
                for nc in range(cand + 1, cand + 25):
                    nd = {nc - a for a in row}
                    nd.add(nc - cand)
                    if not (nd & (used | new)):
                        next_count += 1
                    if next_count >= 4:
                        break
                score = next_count * 100 - cand
                if score > best_score:
                    best_score = score
                    best_cand = cand
            row.append(best_cand)
            used |= {best_cand - a for a in row[:-1]}
            x = best_cand
    return row


def _construct_diverse(n_rows, k, rng, strategy):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, strategy)
        rows.append(row)
    return rows


def _rebuild_row(k, forbidden, rng, max_val, diversify=1):
    """Build a row of k+1 increasing ints starting at 0, all positive diffs
    avoiding `forbidden` and each other, with max entry <= max_val.
    Returns the row or None. Randomized greedy with a bias toward small values
    and a `diversify` chance to skip the smallest candidate for exploration."""
    row = [0]
    local = set()
    for _ in range(k):
        last = row[-1]
        valid = []
        for x in range(last + 1, max_val + 1):
            nd = {x - a for a in row}
            if nd & local or nd & forbidden:
                continue
            valid.append((x, nd))
            if len(valid) >= 6:
                break
        if not valid:
            return None
        # Bias toward small x: pick smallest with high probability, else random.
        r = rng.random()
        if r < 0.65 or len(valid) == 1:
            x, nd = valid[0]
        else:
            x, nd = valid[rng.randint(0, len(valid) - 1)]
        row.append(x)
        local |= nd
    return row


def _try_fit(rows, n_rows, k, rng, target, max_iters=4000):
    """Try to make every row have max entry <= target using row-rebuild moves.
    Returns (rows, scope) where scope is the global max after the attempt."""
    rows = [r[:] for r in rows]
    current_max = _scope(rows)

    for _ in range(max_iters):
        if _time_left() < 5.0:
            break
        if current_max <= target:
            break
        # Pick a row that is above target (the worst one).
        bad = [i for i in range(n_rows) if rows[i][-1] > target]
        if not bad:
            break
        i = max(bad, key=lambda j: rows[j][-1])
        old = rows[i]
        used_other = _all_diffs_except(rows, i)
        new = _rebuild_row(k, used_other, rng, target)
        if new is not None and max(new) < max(old):
            rows[i] = new
            current_max = _scope(rows)
        elif new is not None and max(new) <= max(old):
            # Neutral move: reshuffle to escape local minima.
            if rng.random() < 0.3:
                rows[i] = new
    return rows, current_max


def _local_search(rows, n_rows, k, rng, max_iters=50000):
    """Direct polish: reduce individual entries of the max row and others."""
    best_scope = _scope(rows)
    rows = [r[:] for r in rows]
    iters = 0
    while iters < max_iters and _time_left() > 5.0:
        iters += 1
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        target = best_scope - 1
        if target < k + 1:
            break
        used_other = _all_diffs_except(rows, max_idx)
        row = rows[max_idx]
        improved = False
        for new_last in range(current_max - 1, row[-2], -1):
            if new_last <= row[-2]:
                break
            new_row = row[:-1] + [new_last]
            internal = _row_diffs(new_row)
            if len(internal) != len(new_row) * (len(new_row) - 1) // 2:
                continue
            if internal & used_other:
                continue
            rows[max_idx] = new_row
            new_scope = _scope(rows)
            if new_scope < best_scope:
                best_scope = new_scope
            improved = True
            break
        if not improved:
            found_move = False
            for pos in range(len(row) - 2, 0, -1):
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1
                upper_bound = row[pos + 1] - 1
                for new_val in range(old_val - 1, lower_bound - 1, -1):
                    if new_val < lower_bound or new_val >= upper_bound:
                        continue
                    new_row = row[:pos] + [new_val] + row[pos + 1:]
                    if any(new_row[i] >= new_row[i + 1] for i in range(len(new_row) - 1)):
                        continue
                    internal = _row_diffs(new_row)
                    if len(internal) != len(new_row) * (len(new_row) - 1) // 2:
                        continue
                    if internal & used_other:
                        continue
                    rows[max_idx] = new_row
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                    found_move = True
                    break
                if found_move:
                    break
            if not found_move and rng.random() < 0.5:
                alt_idx = rng.randint(0, n_rows - 1)
                if alt_idx != max_idx:
                    used_other_alt = _all_diffs_except(rows, alt_idx)
                    row_alt = rows[alt_idx]
                    for new_last in range(row_alt[-1] - 1, row_alt[-2], -1):
                        if new_last <= row_alt[-2]:
                            break
                        new_row_alt = row_alt[:-1] + [new_last]
                        internal_alt = _row_diffs(new_row_alt)
                        if len(internal_alt) != len(new_row_alt) * (len(new_row_alt) - 1) // 2:
                            continue
                        if internal_alt & used_other_alt:
                            continue
                        rows[alt_idx] = new_row_alt
                        new_scope = _scope(rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                        break
    return rows, best_scope


def construct(n_rows, k):
    global _DEADLINE, _DEADLINE_SET
    _DEADLINE = time.time() + _TIME_LIMIT
    _DEADLINE_SET = True

    rng = random.Random(12345)

    # Phase 1: diverse randomized initial constructions.
    strategies = ["pure", "shuffled", "lookahead", "shuffled", "lookahead", "pure"]
    best_rows = None
    best_scope = float('inf')
    for strategy in strategies:
        if _time_left() < 25.0:
            break
        rows = _construct_diverse(n_rows, k, rng, strategy)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]

    attempts = 0
    while _time_left() > 35.0 and attempts < 30:
        attempts += 1
        strategy = rng.choice(["pure", "shuffled", "lookahead"])
        rows = _construct_diverse(n_rows, k, rng, strategy)
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
            if best_scope <= 112:
                break

    # Phase 2: binary-search tightening via row-rebuild moves (structural core).
    # Try to fit every row under a shrinking target scope.
    low, high = k + 1, best_scope
    while _time_left() > 25.0 and low < high:
        mid = (low + high) // 2
        if _try_fit(best_rows, n_rows, k, rng, mid, max_iters=2000)[1] <= mid:
            high = mid
        else:
            low = mid + 1
    # Rebuild the best solution under the tightest feasible scope found.
    fit_rows, fit_scope = _try_fit(best_rows, n_rows, k, rng, high, max_iters=3000)
    if fit_scope < best_scope:
        best_scope = fit_scope
        best_rows = fit_rows

    # Phase 3: direct local-search polish.
    if _time_left() > 15.0:
        rows_ls, scope_ls = _local_search(best_rows, n_rows, k, rng, max_iters=40000)
        if scope_ls < best_scope:
            best_scope = scope_ls
            best_rows = rows_ls

    # Phase 4: final tightening pass if time allows.
    if _time_left() > 15.0:
        target = max(k + 1, best_scope - 1)
        fit_rows2, fit_scope2 = _try_fit(best_rows, n_rows, k, rng, target, max_iters=3000)
        if fit_scope2 < best_scope:
            best_scope = fit_scope2
            best_rows = fit_rows2
        rows_ls2, scope_ls2 = _local_search(best_rows, n_rows, k, rng, max_iters=20000)
        if scope_ls2 < best_scope:
            best_scope = scope_ls2
            best_rows = rows_ls2

    # Final validation with a robust fallback.
    if not _validate(best_rows):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = _greedy_row(used, k, None, "pure")
            rows.append(row)
        return rows

    return best_rows
