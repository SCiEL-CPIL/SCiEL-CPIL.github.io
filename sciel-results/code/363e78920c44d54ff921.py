import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start = time.time()
    deadline = start + 115  # safety margin before 120s kill

    # Fast greedy initialization
    rows = _fast_greedy(n_rows, k)
    if not _valid(rows):
        rows = _greedy_with_backtrack(n_rows, k, deadline)

    # Simulated annealing optimization
    rows = _sa_optimize(rows, n_rows, k, deadline)

    # Final local polish
    rows = _local_polish(rows, n_rows, k, deadline)

    return rows


def _fast_greedy(n_rows, k):
    """Greedy construction using difference sets for speed."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        diffs = set()
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _greedy_with_backtrack(n_rows, k, deadline):
    """Greedy with limited backtracking if fast greedy fails."""
    used = set()
    rows = []
    for r in range(n_rows):
        row = [0]
        diffs = set()
        x = 0
        while len(row) < k + 1 and time.time() < deadline - 5:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used |= new_diffs
        if len(row) < k + 1:
            # Backtrack: rebuild this row with a different strategy
            row = [0]
            diffs = set()
            x = 0
            while len(row) < k + 1 and time.time() < deadline - 5:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
        rows.append(row)
    return rows


def _valid(rows):
    """Check if all pairwise differences across all rows are distinct."""
    seen = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _scope(rows):
    """Return the maximum entry across all rows."""
    return max(max(row) for row in rows)


def _row_diffs(row):
    """Compute all pairwise differences for a single row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _sa_optimize(rows, n_rows, k, deadline):
    """Simulated annealing to minimize scope."""
    import math

    best = [list(r) for r in rows]
    best_scope = _scope(best)
    current = [list(r) for r in rows]
    current_scope = best_scope

    rng = random.Random(12345)
    t0 = time.time()
    initial_temp = max(1.0, (best_scope - 100) * 0.1)
    cooling = 0.9995

    # Precompute used diffs for fast validity check
    def _all_diffs(rows):
        s = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    s.add(row[j] - row[i])
        return s

    all_diffs = _all_diffs(current)
    total_diffs = len(all_diffs)

    while time.time() < deadline - 10:
        temp = initial_temp * (cooling ** (int((time.time() - t0) * 100)))
        if temp < 0.01:
            temp = 0.01

        # Pick a random row and position to modify
        ri = rng.randrange(n_rows)
        row = current[ri]
        pos = rng.randrange(1, len(row))  # don't touch position 0

        old_val = row[pos]
        # Try several candidate new values
        candidates = []
        for delta in range(-8, 9):
            nv = old_val + delta
            if nv <= 0:
                continue
            if pos > 0 and nv <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and nv >= row[pos + 1]:
                continue
            candidates.append(nv)

        if not candidates:
            continue

        new_val = rng.choice(candidates)
        row[pos] = new_val

        # Check validity incrementally
        # Remove old diffs involving position pos
        old_diffs = set()
        for i in range(len(row)):
            if i == pos:
                continue
            d = abs(row[pos] - row[i])
            old_diffs.add(d)

        # Actually, let's just recompute for simplicity but check efficiently
        new_all = set()
        valid = True
        for r_idx, r in enumerate(current):
            n = len(r)
            for i in range(n):
                for j in range(i + 1, n):
                    d = r[j] - r[i]
                    if d in new_all:
                        valid = False
                        break
                    new_all.add(d)
            if not valid:
                break

        if not valid:
            row[pos] = old_val
            continue

        new_scope = _scope(current)
        delta_scope = new_scope - current_scope

        if delta_scope <= 0 or rng.random() < math.exp(-delta_scope / temp):
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best = [list(r) for r in current]
        else:
            row[pos] = old_val

    return best


def _local_polish(rows, n_rows, k, deadline):
    """Final local search: try to reduce scope by adjusting individual elements."""
    best = [list(r) for r in rows]
    best_scope = _scope(best)

    def _check_valid(rows):
        seen = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for ri in range(n_rows):
            row = best[ri]
            for pos in range(1, len(row)):
                old_val = row[pos]
                # Try to decrease the value
                for nv in range(old_val - 1, 0, -1):
                    if pos > 0 and nv <= row[pos - 1]:
                        break
                    if pos < len(row) - 1 and nv >= row[pos + 1]:
                        continue
                    row[pos] = nv
                    if _check_valid(best):
                        new_scope = _scope(best)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            improved = True
                        else:
                            row[pos] = old_val
                        break
                    else:
                        row[pos] = old_val

    return best

# EVOLVE-BLOCK-END
