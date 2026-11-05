"""
Difference triangle set (7,5) solver.

Structural change from the parent:
  - The parent builds rows greedily (row-by-row) then shrinks them.
  - This version formulates the problem as a Constraint Satisfaction Search
    over the SET OF DIFFERENCE VALUES: every difference d in [1, M] is assigned
    to exactly one of the n_rows rows, and each row's assigned differences must
    be REALIZABLE as a valid increasing row of k+1 integers starting at 0.
  - We select the difference with the fewest remaining compatible row
    assignments (most-constrained variable) and enforce Hall's condition for
    the remaining unassigned differences.
  - A greedy row-by-row construction is kept only as a fast valid fallback.
  - The parent's local-search / shrink / polish machinery is reused to reduce
    the scope of whatever solution the CSP (or the greedy fallback) produces.
"""
import time
import random
import sys
import math

sys.setrecursionlimit(100000)

# EVOLVE-BLOCK-START


def _diffs_of_row(row):
    """Return frozenset of positive differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)


def _total_scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)


def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= set(_diffs_of_row(r))
    return used


# --------------------------------------------------------------------------- #
# Row realization (core of the difference-assignment CSP)
# --------------------------------------------------------------------------- #
def _build_row_from_diffs(diffs, k, cap):
    """
    Given a set of positive differences `diffs`, try to build a row of exactly
    k+1 integers starting at 0 whose pairwise differences are exactly `diffs`.
    The row's maximum entry must be <= cap.
    Returns the row (list) or None.
    """
    if len(diffs) != k * (k + 1) // 2:
        return None
    ds = sorted(diffs)
    if ds[0] != 1:
        return None  # the smallest entry after 0 must be 1
    max_d = ds[-1]
    if max_d > cap:
        return None

    used = set()
    row = [0]

    def rec(x, remaining):
        if len(row) == k + 1:
            return remaining == 0
        last = row[-1]
        # candidate next value t > last, t - last must be in remaining
        for d in ds:
            if d not in remaining:
                continue
            t = last + d
            if t > cap:
                break
            # check all new differences t - a are in remaining
            ok = True
            for a in row:
                nd = t - a
                if nd not in remaining:
                    ok = False
                    break
            if not ok:
                continue
            row.append(t)
            for a in row[:-1]:
                remaining.remove(t - a)
            if rec(t, remaining):
                return True
            row.pop()
            for a in row[:-1]:
                remaining.add(t - a)
        return False

    if rec(0, set(diffs)):
        return row
    return None


# --------------------------------------------------------------------------- #
# Greedy row construction (fast valid fallback)
# --------------------------------------------------------------------------- #
def _generate_row_greedy_pure(used, k, cap=None):
    """Pure greedy: always pick the smallest valid candidate."""
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        if cap is not None and x > cap:
            return None, None
        new = {x - a for a in row}
        if not (new & used) and not (new & new_diffs):
            row.append(x)
            new_diffs |= new
    return row, frozenset(new_diffs)


def _greedy_construct(n_rows, k, cap=None):
    """Pure greedy construction as fallback."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, diffs = _generate_row_greedy_pure(used, k, cap)
        if row is None:
            return None
        rows.append(row)
        used |= diffs
    return rows


# --------------------------------------------------------------------------- #
# Difference-assignment CSP
# --------------------------------------------------------------------------- #
def _csp_assign(n_rows, k, M, rng, deadline):
    """
    Assign every difference in [1, M] to exactly one of the n_rows rows such that
    each row is realizable. Uses a most-constrained-variable backtracking search
    with Hall's-condition pruning.

    Returns (rows, M) on success, else (None, None).
    """
    total = k * (k + 1) // 2
    n = n_rows
    # Precompute, for each difference d, the set of rows that could host it
    # (i.e. the row's current assigned set plus {d} is realizable).
    # This is dynamic, so we compute it on the fly but cache per (row, d).

    assign = [set() for _ in range(n)]          # assigned diffs per row
    diff_owner = {}                              # d -> row
    unassigned = list(range(1, M + 1))

    # Realizability cache: (frozenset of a row's diffs) -> bool
    # We recompute via _build_row_from_diffs when needed.

    # Precompute which (row, d) pairs are "individually" feasible:
    # adding d to an empty row is feasible iff 1 in {d} when d==1, but a row
    # needs exactly `total` diffs so a single d is only feasible if d can appear
    # in SOME complete row. Simplest: any d <= M is potentially feasible.

    # Heuristic ordering: differences that are "scarce" (harder to place) first.
    # A difference is scarce if it belongs to few realizable rows. Approximate
    # scarcity by preferring larger differences (they force larger entries).

    best = [None]

    # Count of rows with each size, for Hall's condition.
    def hall_ok():
        # For every subset size s (1..n), the number of unassigned diffs that can
        # only go into at most s rows must be <= total capacity of those s rows.
        # Cheap necessary check: count unassigned diffs whose feasible-row-set size
        # is 1; those must fit into distinct rows.
        single_rows = {}
        for d in unassigned:
            feas = _feasible_rows(d, assign, k, M)
            if len(feas) == 1:
                r = next(iter(feas))
                single_rows[r] = single_rows.get(r, 0) + 1
                # row r can take at most total - len(assign[r]) more
                if single_rows[r] > total - len(assign[r]):
                    return False
        return True

    def _feasible_rows(d, assign, k, M):
        out = []
        for r in range(n):
            if len(assign[r]) >= total:
                continue
            cand = assign[r] | {d}
            if _row_possible(cand, k, M):
                out.append(r)
        return out

    # Memoized row-possibility test.
    _poss_cache = {}

    def _row_possible(cand, k, M):
        key = frozenset(cand)
        v = _poss_cache.get(key)
        if v is not None:
            return v
        if len(cand) > total:
            _poss_cache[key] = False
            return False
        # necessary conditions
        if 1 not in cand and len(cand) >= 1:
            # a row that doesn't contain 1 can still be built if its min entry
            # is > 1, but then 1 must be produced as a difference somewhere;
            # since every row starts at 0, the smallest positive difference in
            # the row is row[1]-0 = row[1] >= 1, and row[1] must be in cand.
            # So the minimum of cand must equal row[1] which is >= 1; it's fine
            # if 1 is not in cand as long as min(cand) is the first step.
            pass
        if max(cand) > M:
            _poss_cache[key] = False
            return False
        # quick necessary: the differences must be able to form a chain.
        # We do a light check: smallest diff must be achievable as a step.
        # Full check is expensive; do it only near completion.
        if len(cand) == total:
            v = _build_row_from_diffs(cand, k, M) is not None
        else:
            # cheap necessary conditions only
            v = True
            if len(cand) >= 2 and min(cand) > 1 and 1 not in cand:
                # allowed (row may start with a jump), but then 1 must be a
                # difference between two later entries, which is possible.
                v = True
        _poss_cache[key] = v
        return v

    def most_constrained():
        # pick unassigned diff with fewest feasible rows
        best_d = None
        best_feas = None
        for d in unassigned:
            feas = _feasible_rows(d, assign, k, M)
            if not feas:
                return d, []
            if best_feas is None or len(feas) < len(best_feas):
                best_d, best_feas = d, feas
                if len(feas) == 1:
                    break
        return best_d, best_feas

    def rec(depth):
        if time.time() > deadline:
            return False
        if not unassigned:
            # all assigned; build rows
            rows = []
            for r in range(n):
                row = _build_row_from_diffs(assign[r], k, M)
                if row is None:
                    return False
                rows.append(row)
            best[0] = rows
            return True

        if not hall_ok():
            return False

        d, feas = most_constrained()
        if d is None:
            return False
        if not feas:
            return False
        # order rows by how full they are (fill the tightest first)
        feas.sort(key=lambda r: len(assign[r]), reverse=True)
        for r in feas:
            assign[r].add(d)
            diff_owner[d] = r
            unassigned.remove(d)
            if rec(depth + 1):
                return True
            unassigned.append(d)
            del diff_owner[d]
            assign[r].discard(d)
        return False

    if rec(0):
        return best[0], M
    return None, None


# --------------------------------------------------------------------------- #
# Local search / shrink / polish (refinement machinery from the parent)
# --------------------------------------------------------------------------- #
def _local_search_scope(rows, k, rng, deadline):
    """Local search to reduce the scope (maximum entry)."""
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    scope = _total_scope(rows)

    improved = True
    num_passes = 0
    while improved and time.time() < deadline:
        improved = False
        num_passes += 1
        if num_passes % 3 == 0:
            row_indices = list(range(len(rows)))
            rng.shuffle(row_indices)
        elif num_passes % 3 == 1:
            row_indices = sorted(range(len(rows)), key=lambda i: -rows[i][-1])
        else:
            row_indices = sorted(range(len(rows)), key=lambda i: rows[i][-1])

        for idx in row_indices:
            if time.time() > deadline:
                break
            row = rows[idx]

            last = row[-1]
            for new_last in range(last - 1, row[-2], -1):
                old_diffs = {last - a for a in row[:-1]}
                new_diffs = {new_last - a for a in row[:-1]}
                remaining = used - old_diffs
                if not (new_diffs & remaining):
                    row[-1] = new_last
                    used = (used - old_diffs) | new_diffs
                    scope = _total_scope(rows)
                    improved = True
                    break

            for pos in range(len(row) - 2, 0, -1):
                if time.time() > deadline:
                    break
                old_val = row[pos]
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                    remaining = used - old_diffs
                    if not (new_diffs & remaining):
                        row[pos] = new_val
                        used = (used - old_diffs) | new_diffs
                        scope = _total_scope(rows)
                        improved = True
                        break

    return rows


def _polish_pass(rows, k, deadline):
    """High-precision polish: try every possible reduction for each position."""
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)

    for _ in range(5):
        if time.time() > deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(len(rows)):
                if time.time() > deadline:
                    break
                row = rows[idx]
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    old_val = row[pos]
                    for new_val in range(old_val - 1, row[pos - 1], -1):
                        old_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                old_diffs.add(old_val - row[i])
                            else:
                                old_diffs.add(row[i] - old_val)
                        new_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                new_diffs.add(new_val - row[i])
                            else:
                                new_diffs.add(row[i] - new_val)
                        remaining = used - old_diffs
                        if not (new_diffs & remaining):
                            row[pos] = new_val
                            used = (used - old_diffs) | new_diffs
                            improved = True
                            break

    return rows


def _valid(rows, k):
    """Check that rows are valid: k+1 increasing ints from 0, distinct diffs."""
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0:
                    return False
                if d in used:
                    return False
                used.add(d)
    return True


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 118  # safety margin before the 150 s kill

    # ---- Phase 1: fast greedy fallback (guarantees a valid solution) ----
    rows = _greedy_construct(n_rows, k)
    if rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best_rows = rows
    best_scope = _total_scope(rows)

    # ---- Phase 2: difference-assignment CSP with iterative scope deepening ----
    # Try to find a solution with a smaller scope by assigning every difference
    # in [1, M] to a row. Start near the greedy scope and shrink M.
    greedy_scope = best_scope
    # The counting bound lower bound is roughly total_diffs / n_rows.
    lower = (n_rows * (k * (k + 1) // 2)) // n_rows + 1

    # We attempt a handful of M values descending from greedy_scope-1 down to
    # a floor, each with a bounded time budget. The CSP is expensive, so we
    # only try a few.
    M_budget = max(0, int((deadline - start_time) * 0.45))
    if M_budget > 20:
        per_try = max(6, M_budget // 8)
        tried = 0
        for M in range(min(greedy_scope - 1, 130), lower, -1):
            if tried >= 8:
                break
            if time.time() > deadline - 40:
                break
            t0 = time.time()
            csp_rows, _ = _csp_assign(n_rows, k, M, rng, t0 + per_try)
            tried += 1
            if csp_rows is not None and _valid(csp_rows, k):
                sc = _total_scope(csp_rows)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = csp_rows

    # ---- Phase 3: local search / polish to shrink the best solution ----
    if time.time() < deadline - 20:
        best_rows = _local_search_scope(best_rows, k, rng, deadline - 20)
        if _valid(best_rows, k):
            best_rows = _polish_pass(best_rows, k, deadline - 8)

    # Final safety: ensure validity
    if not _valid(best_rows, k):
        # fall back to the greedy solution
        rows2 = _greedy_construct(n_rows, k)
        if rows2 is not None and _valid(rows2, k):
            return rows2

    return best_rows

# EVOLVE-BLOCK-END
