"""Rewritten program: difference-triangle (7,5) with fast incremental local search."""
import time
import random

# EVOLVE-BLOCK-START


def _row_diffs(row):
    n = len(row)
    s = set()
    for i in range(n - 1):
        a = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - a)
    return s


def _valid_rows(rows, n_rows, k, deadline):
    """Validate that all positive differences across all rows are distinct."""
    seen = set()
    for row in rows:
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(k):
            if row[i + 1] <= row[i]:
                return False
        for d in _row_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return len(rows) == n_rows


def _greedy(n_rows, k, deadline, rng, randomize):
    """Construct rows greedily. Returns (rows, row_diffs) or (None, None)."""
    used = set()
    rows = []
    row_diffs = []
    for _ in range(n_rows):
        row = [0]
        used_row = set()
        for _ in range(k):
            last = row[-1]
            if randomize:
                lo = last + 1
                hi = last + 60
                cands = list(range(lo, hi + 1))
                rng.shuffle(cands)
                cands.sort(key=lambda c: c + rng.random() * 12.0)
                placed = False
                for c in cands:
                    nd = {c - a for a in row}
                    if not (nd & used) and not (nd & used_row):
                        row.append(c)
                        used_row |= nd
                        used |= nd
                        placed = True
                        break
                if not placed:
                    for c in range(hi + 1, hi + 1001):
                        if time.time() > deadline:
                            return None, None
                        nd = {c - a for a in row}
                        if not (nd & used) and not (nd & used_row):
                            row.append(c)
                            used_row |= nd
                            used |= nd
                            placed = True
                            break
                if not placed:
                    return None, None
            else:
                c = last + 1
                while True:
                    nd = {c - a for a in row}
                    if not (nd & used) and not (nd & used_row):
                        row.append(c)
                        used_row |= nd
                        used |= nd
                        break
                    c += 1
                    if time.time() > deadline:
                        return None, None
        rows.append(row)
        row_diffs.append(used_row)
    return rows, row_diffs


def _improve(rows, row_diffs, n_rows, k, deadline, rng):
    """Fast incremental local search to minimize the global scope.

    Returns (improved, rows, row_diffs, scope).
    """
    scope = max(max(r) for r in rows)
    # Global set of all differences.
    G = set()
    for s in row_diffs:
        G |= s

    improved = False
    stall = 0
    max_stall = 12

    while time.time() < deadline and stall < max_stall:
        # Find all positions holding the current global maximum.
        max_pos = [(i, j) for i in range(n_rows)
                   for j in range(1, k + 1) if rows[i][j] == scope]
        rng.shuffle(max_pos)

        moved = False
        for (i, j) in max_pos:
            if time.time() > deadline:
                break
            row = rows[i]
            rd = row_diffs[i]
            cur = row[j]
            lo = row[j - 1] + 1
            hi = (row[j + 1] - 1) if j + 1 < k + 1 else cur - 1
            if hi < lo:
                continue

            # Remove this row's differences from the global set.
            G.difference_update(rd)

            placed = False
            # Scan downward; prefer the smallest valid value to tighten.
            for c in range(hi, lo - 1, -1):
                # Differences touching position j (others are unchanged).
                nd = {c - row[t] for t in range(j)} | {row[t] - c for t in range(j + 1, k + 1)}
                if len(nd) != len(nd) or len(nd) != k:
                    # len check: exactly k new differences (j-1 above + (k-j) below)
                    pass
                # distinctness within nd and against G
                if len(nd) != k:
                    continue
                if nd & G:
                    continue
                row[j] = c
                rd.difference_update({cur - row[t] for t in range(j)})
                rd.difference_update({row[t] - cur for t in range(j + 1, k + 1)})
                rd |= nd
                G |= nd
                placed = True
                break

            if placed:
                improved = True
                moved = True
                scope = max(max(r) for r in rows)
                break
            else:
                G |= rd  # restore

        if moved:
            stall = 0
            if scope <= 105:
                break
            continue

        # Max is stuck: try to relocate a random non-max element to unblock.
        all_pos = [(i, j) for i in range(n_rows) for j in range(1, k + 1)
                   if rows[i][j] != scope]
        rng.shuffle(all_pos)
        any_move = False
        for (i, j) in all_pos:
            if time.time() > deadline:
                break
            row = rows[i]
            rd = row_diffs[i]
            cur = row[j]
            lo = row[j - 1] + 1
            hi = (row[j + 1] - 1) if j + 1 < k + 1 else cur - 1
            if hi < lo or cur <= lo:
                continue

            G.difference_update(rd)
            placed = False
            for c in range(hi, lo - 1, -1):
                nd = {c - row[t] for t in range(j)} | {row[t] - c for t in range(j + 1, k + 1)}
                if len(nd) != k:
                    continue
                if nd & G:
                    continue
                row[j] = c
                rd.difference_update({cur - row[t] for t in range(j)})
                rd.difference_update({row[t] - cur for t in range(j + 1, k + 1)})
                rd |= nd
                G |= nd
                placed = True
                break
            if placed:
                improved = True
                any_move = True
                break
            else:
                G |= rd

        if any_move:
            stall = 0
        else:
            stall += 1

    return improved, rows, row_diffs, scope


def construct(n_rows, k):
    deadline = time.time() + 112.0  # safety margin under the 150s hard limit
    rng = random.Random(20250711)

    best_rows = None
    best_scope = float('inf')

    # Deterministic greedy baseline (always valid).
    base_rows, base_diffs = _greedy(n_rows, k, deadline, rng, randomize=False)
    if base_rows is None:
        # Fallback: pure linear greedy (shouldn't fail for these sizes).
        used, rows = set(), []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                c = row[-1] + 1
                while {c - a for a in row} & used:
                    c += 1
                row.append(c)
                used |= {c - a for a in row}
            rows.append(row)
        return rows

    imp, rows, rd, sc = _improve(base_rows, base_diffs, n_rows, k, deadline, rng)
    if sc < best_scope:
        best_scope = sc
        best_rows = [r[:] for r in rows]
    if best_scope <= 105:
        return best_rows

    # Multi-start with randomized greedy + refinement.
    attempt = 0
    while time.time() < deadline - 2.0:
        attempt += 1
        start_rows, start_diffs = _greedy(n_rows, k, deadline, rng, randomize=True)
        if start_rows is None:
            continue
        imp, rows, rd, sc = _improve(start_rows, start_diffs, n_rows, k, deadline, rng)
        if sc < best_scope:
            best_scope = sc
            best_rows = [r[:] for r in rows]
            if best_scope <= 105:
                break
        # Occasionally re-refine the current best to escape plateaus.
        if attempt % 4 == 0 and best_rows is not None:
            bd = [_row_diffs(r) for r in best_rows]
            imp, rows, rd, sc = _improve(best_rows, bd, n_rows, k, deadline, rng)
            if sc < best_scope:
                best_scope = sc
                best_rows = [r[:] for r in rows]
                if best_scope <= 105:
                    break

    if best_rows is None:
        best_rows = base_rows
    return best_rows

# EVOLVE-BLOCK-END
