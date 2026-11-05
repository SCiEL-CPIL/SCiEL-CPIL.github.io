import time
import random

# EVOLVE-BLOCK-START
def _greedy_seed(n_rows, k, rng):
    """Greedy ruler construction: each row takes the smallest next mark whose
    differences to earlier marks are all unused."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _max_scope(rows):
    return max(r[-1] for r in rows)


def _diffs_of_row(row):
    return [row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row))]


def _is_valid(rows):
    seen = set()
    for r in rows:
        if r[0] != 0:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _greedy_rebuild_row(rows, idx, k, rng):
    """Rebuild row idx from scratch with greedy, keeping other rows' diffs used."""
    others = [r for i, r in enumerate(rows) if i != idx]
    used = set()
    for r in others:
        used.update(_diffs_of_row(r))
    row, x = [0], 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used):
            row.append(x)
            used |= new
    return row


def _perturb_mark(rows, idx, rng):
    """Move a single non-zero mark of row idx to a nearby value (staying in range)."""
    row = rows[idx]
    n = len(row)
    # pick a mark index 1..n-1
    mi = rng.randint(1, n - 1)
    lo = row[mi - 1] + 1
    hi = row[mi + 1] - 1 if mi < n - 1 else row[-1] + 3
    if lo > hi:
        hi = lo
    new_val = rng.randint(lo, hi)
    if new_val == row[mi]:
        return False
    row[mi] = new_val
    return True


def _perturb_last(rows, idx, rng):
    """Change the last (largest) mark of row idx to shrink or grow the scope."""
    row = rows[idx]
    n = len(row)
    if n < 2:
        return False
    lo = row[n - 2] + 1
    hi = row[n - 1] + 4
    new_val = rng.randint(lo, hi)
    if new_val == row[n - 1]:
        return False
    row[n - 1] = new_val
    return True


def construct(n_rows, k):
    start = time.time()
    deadline = start + 140.0  # safety margin below 150s
    rng = random.Random(12345)

    # Seed with greedy
    rows = _greedy_seed(n_rows, k, rng)
    if not _is_valid(rows):
        # Fallback: return greedy even if (shouldn't happen)
        return rows

    best = [r[:] for r in rows]
    best_score = _max_scope(best)

    # Simulated annealing
    T = 50.0
    T_min = 0.01
    cool = 0.9990
    n = k + 1

    while time.time() < deadline and T > T_min:
        # Decide perturbation type
        r = rng.random()
        idx = rng.randint(0, n_rows - 1)
        if r < 0.12:
            # multi-row rebuild
            rows[idx] = _greedy_rebuild_row(rows, idx, k, rng)
            if n_rows > 1 and rng.random() < 0.5:
                idx2 = rng.randint(0, n_rows - 1)
                if idx2 != idx:
                    rows[idx2] = _greedy_rebuild_row(rows, idx2, k, rng)
        elif r < 0.45:
            _perturb_last(rows, idx, rng)
        else:
            _perturb_mark(rows, idx, rng)

        if not _is_valid(rows):
            # reject invalid
            T *= cool
            continue

        score = _max_scope(rows)
        delta = score - best_score
        if delta <= 0:
            best = [r[:] for r in rows]
            best_score = score
        else:
            # accept with probability exp(-delta/T)
            if delta < 500 and rng.random() < pow(2.718281828459045, -delta / max(T, 1e-9)):
                pass  # keep perturbed rows as current
            else:
                # revert: rebuild rows to best
                rows = [r[:] for r in best]
        T *= cool

    # Final polish: accept only improvements, tighter
    while time.time() < deadline - 1.0:
        idx = rng.randint(0, n_rows - 1)
        r = rng.random()
        if r < 0.5:
            _perturb_last(rows, idx, rng)
        else:
            _perturb_mark(rows, idx, rng)
        if _is_valid(rows):
            score = _max_scope(rows)
            if score < best_score:
                best = [r[:] for r in rows]
                best_score = score
            else:
                rows = [r[:] for r in best]
        else:
            rows = [r[:] for r in best]

    return best
# EVOLVE-BLOCK-END
