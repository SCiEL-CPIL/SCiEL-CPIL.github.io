"""Difference triangle set (4,7) – randomized greedy + swap local search with deadline guard."""
import time
import random
from itertools import combinations

_DEADLINE = 150.0  # evaluator kills at 150s; keep a safety margin
_MARGIN = 5.0      # seconds to keep before the hard limit


def _row_diffs(row):
    """Return the set of positive differences a[j]-a[i] for i<j in a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            diffs.add(row[j] - ai)
    return diffs


def _all_diffs(rows):
    """Return the union of all positive differences across all rows."""
    used = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            ai = r[i]
            for j in range(i + 1, n):
                used.add(r[j] - ai)
    return used


def _valid(rows, k):
    """Check that rows form a valid difference triangle set."""
    if len(rows) != 4:
        return False
    for r in rows:
        if len(r) != k + 1:
            return False
        if r[0] != 0:
            return False
        for i in range(len(r) - 1):
            if r[i + 1] <= r[i]:
                return False
    used = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            ai = r[i]
            for j in range(i + 1, n):
                d = r[j] - ai
                if d in used:
                    return False
                used.add(d)
    return True


def _scope(rows):
    """The largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_row(used, k, rng):
    """Build one row of k+1 increasing integers starting at 0, using the smallest
    next mark whose new differences don't conflict with `used`.
    With rng, we occasionally skip a candidate to add diversity."""
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not (new_diffs & used):
            # With small probability, skip this candidate to explore alternatives
            if rng.random() < 0.05 and x < 200:
                continue
            row.append(x)
            used |= new_diffs
    return row


def _greedy_construct(n_rows, k, rng):
    """Build n_rows rows using randomized greedy."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng)
        rows.append(row)
    return rows


def _swap_local_search(rows, k, rng, deadline):
    """Try to reduce the scope by swapping a mark in a row with a smaller valid one.
    Returns the improved rows or the original if no improvement found."""
    n_rows = len(rows)
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    used = _all_diffs(best)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize row and position order
        positions = []
        for ri in range(n_rows):
            for pi in range(1, len(best[ri])):  # skip index 0 (always 0)
                positions.append((ri, pi))
        rng.shuffle(positions)
        for ri, pi in positions:
            if time.time() >= deadline:
                break
            row = best[ri]
            old_val = row[pi]
            # Try replacing old_val with a smaller value that maintains increasing order
            lower = row[pi - 1] + 1
            upper = row[pi + 1] - 1 if pi + 1 < len(row) else old_val - 1
            if upper < lower:
                continue
            # Try values in random order, preferring smaller
            candidates = list(range(upper, lower - 1, -1))
            rng.shuffle(candidates)
            for new_val in candidates:
                if new_val >= old_val:
                    continue
                # Check increasing order
                if pi > 0 and new_val <= row[pi - 1]:
                    break
                if pi + 1 < len(row) and new_val >= row[pi + 1]:
                    continue
                # Compute new differences for this row
                # Remove old diffs involving position pi
                old_diffs = set()
                for i in range(pi):
                    old_diffs.add(old_val - row[i])
                for j in range(pi + 1, len(row)):
                    old_diffs.add(row[j] - old_val)
                # Add new diffs
                new_diffs = set()
                for i in range(pi):
                    new_diffs.add(new_val - row[i])
                for j in range(pi + 1, len(row)):
                    new_diffs.add(row[j] - new_val)
                # Check no conflict with global used (minus old_diffs)
                remaining_used = used - old_diffs
                if new_diffs & remaining_used:
                    continue
                # Apply the swap
                row[pi] = new_val
                used = (used - old_diffs) | new_diffs
                new_scope = _scope(best)
                if new_scope < best_scope:
                    best_scope = new_scope
                    improved = True
                    break
                else:
                    # Revert
                    row[pi] = old_val
                    used = (used - new_diffs) | old_diffs
    return best, best_scope


def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences across all rows must be distinct.
    Minimize the largest entry (scope)."""
    start_time = time.time()
    deadline = start_time + _DEADLINE - _MARGIN

    # Quick check for small cases
    if n_rows == 0:
        return []

    # Start with a deterministic greedy to have a baseline
    rng_base = random.Random(42)
    baseline = _greedy_construct(n_rows, k, rng_base)
    best = [r[:] for r in baseline]
    best_scope = _scope(best)

    # Try multiple randomized restarts
    rng = random.Random()
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        # Randomized greedy construction
        candidate = _greedy_construct(n_rows, k, rng)
        cand_scope = _scope(candidate)

        # Local search (swap-based)
        if time.time() < deadline - 2.0:
            candidate, cand_scope = _swap_local_search(
                candidate, k, rng, deadline - 1.0
            )

        if cand_scope < best_scope:
            best = [r[:] for r in candidate]
            best_scope = cand_scope

        # If we've hit the theoretical lower bound, stop
        # Lower bound: need n_rows * C(k+1,2) distinct positive differences,
        # max difference <= scope, so scope >= n_rows * C(k+1,2) / 1 ... 
        # Actually the differences range from 1 to scope, so we need scope >= n_rows * C(k+1,2)
        # But that's a very loose bound. A tighter one: the number of distinct diffs
        # is n_rows * k*(k+1)/2, and they must all be in [1, scope], so scope >= that count.
        min_possible = n_rows * k * (k + 1) // 2
        if best_scope <= min_possible:
            break

        # Occasionally do a more aggressive local search on the best
        if attempt % 3 == 0 and time.time() < deadline - 5.0:
            refined, ref_scope = _swap_local_search(
                [r[:] for r in best], k, rng, deadline - 2.0
            )
            if ref_scope < best_scope:
                best = refined
                best_scope = ref_scope

    return best


# For convenience, also expose a pure greedy and scope helper (matching observed units)
def pure_greedy(n_rows, k):
    """Deterministic greedy construction."""
    rng = random.Random(0)
    return _greedy_construct(n_rows, k, rng)


def randomized_greedy(n_rows, k, seed=None):
    """Randomized greedy construction."""
    rng = random.Random(seed)
    return _greedy_construct(n_rows, k, rng)


def scope_of(rows):
    """Return the scope (largest entry) of a set of rows."""
    return _scope(rows)
