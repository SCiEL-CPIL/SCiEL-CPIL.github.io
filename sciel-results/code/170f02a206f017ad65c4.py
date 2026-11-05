import time
import random
import math
from collections import defaultdict, deque

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    """
    Difference triangle set: n_rows rows of k+1 increasing integers starting at 0,
    all positive within-row differences distinct across all rows.
    Minimize the maximum entry (the scope).
    """
    deadline = time.time() + 145.0  # wall-clock limit is 150s

    # Per-row structure: a row is a SET of k+1 values {0, a1, ..., a_k}.
    # Its differences D = {v - u : v > u in set} are C(k+1,2) = k(k+1)/2 distinct
    # positive integers. Across rows the D-sets must be pairwise disjoint.
    #
    # KEY REFORMULATION: if a row-set has max value M, then every difference in D
    # is <= M, and M itself is always in D (M - 0). So the "top" difference of a row
    # equals its scope contribution. We therefore pack the LARGEST differences first:
    # the hardest constraint (the largest available difference values) binds the scope.

    random.seed(20250711)

    # ---- Phase 1: build a valid baseline with backtracking (largest-diff-first) ----
    # This gives a guaranteed-valid initial scope (upper bound).
    best_rows, best_scope = _backtrack_construct(n_rows, k, deadline, time_budget=10.0)

    if time.time() > deadline - 2.0:
        return best_rows

    # ---- Phase 2: iterative scope reduction (structural search) ----
    # Try to find a valid packing with a smaller scope by treating the problem as
    # a CSP: assign each row a difference-set drawn from [1..scope] such that the
    # sets are disjoint and each set is realizable as a difference set of a (0-inclusive)
    # k+1-subset of [0..scope].
    best_rows, best_scope = _scope_reduction_search(
        n_rows, k, best_rows, best_scope, deadline
    )

    return best_rows


def _diffs_of_set(values):
    """All positive differences of a set of values."""
    vals = sorted(values)
    out = set()
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            out.add(vals[j] - vals[i])
    return out


def _realize_set(diff_set, scope):
    """
    Given a target set of differences D (size k(k+1)/2) with max <= scope,
    try to find a k+1-element subset S of [0..scope] containing 0 such that
    diff set(S) == D. Returns the sorted list S or None.
    """
    # Heuristic reconstruction: 0 in S. The largest element m = max(S) must equal
    # max(D) (since m - 0 = m is a difference and no difference exceeds m).
    if not D_realization_check(diff_set):
        return None
    return _build_from_diffs(diff_set, scope)


def D_realization_check(diff_set):
    # A necessary condition: diff_set must contain its own max, and the difference
    # set of any realizable set is closed under the property that max is present.
    return max(diff_set) in diff_set


def _build_from_diffs(diff_set, scope):
    """
    Reconstruct a value set from its difference set.
    Strategy: the value set S contains 0 and m=max(S)=max(diff_set).
    We build S greedily: repeatedly pick the smallest x in [1..scope] not yet in S
    such that {x - s : s in S, s < x} and {s - x : s in S, s > x} are all in diff_set,
    and don't create a difference outside diff_set.
    """
    m = max(diff_set)
    S = {0, m}
    # The differences that must be accounted for by the final S
    remaining = set(diff_set)
    # differences already produced by {0, m}
    produced = {m}
    remaining -= produced

    target_size = None
    # We don't know k here directly; but |diff_set| = C(k+1,2) so k+1 = n where
    # n(n-1)/2 = len(diff_set). Recover n.
    L = len(diff_set)
    n = None
    for cand in range(2, 200):
        if cand * (cand - 1) // 2 == L:
            n = cand
            break
    if n is None:
        return None

    # Greedy fill: pick next smallest candidate value that is consistent.
    # Candidate values are the "sums" hints: a value x in S implies x = s + d for
    # some s in S and d in diff_set, OR x = m - d for some d in diff_set.
    # Build a frontier of possible values.
    candidates = set()
    for d in diff_set:
        candidates.add(d)        # 0 + d
        candidates.add(m - d)    # m - d
    candidates.discard(0)
    candidates.discard(m)

    for _ in range(n - 2):
        # Try candidates in increasing order
        picked = None
        for x in sorted(candidates):
            if x in S:
                continue
            if x <= 0 or x >= m:
                continue
            # check all differences with current S are in diff_set and in remaining/produced
            ok = True
            new_diffs = set()
            for s in S:
                d = abs(x - s)
                if d not in diff_set:
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                continue
            # differences must not collide with each other (they won't since |S|<n)
            picked = x
            break
        if picked is None:
            return None
        S.add(picked)
        # add new differences
        for s in S:
            if s != picked:
                d = abs(picked - s)
                remaining.discard(d)
        # expand candidates
        for d in diff_set:
            candidates.add(picked + d)
            candidates.add(picked - d)
            candidates.add(m - d)
        candidates.discard(0)
        candidates.discard(m)

    if len(S) != n:
        return None
    # final validation
    if _diffs_of_set(S) != diff_set:
        return None
    return sorted(S)


def _backtrack_construct(n_rows, k, deadline, time_budget):
    """
    Backtracking construction assigning rows one at a time, largest-difference-first.
    We choose each row's difference-set by greedy with backtracking on conflicts.
    Returns (rows, scope).
    """
    used = set()
    rows = []
    # We'll assign rows; each row must get a distinct difference-set.
    # To keep scope small, prefer rows that use smaller maximum differences.
    # But the global scope = max over rows of max(value). For a row with diff-set D,
    # its max value is max(D). So global scope = max over rows of max(D_row).

    # Generate candidate rows greedily (smallest possible max), tracking used diffs.
    # Use a randomized greedy to get a valid solution; then we reduce.
    return _greedy_with_backtrack(n_rows, k, used, rows, deadline, time_budget)


def _greedy_with_backtrack(n_rows, k, used, rows, deadline, time_budget):
    """
    Build rows greedily. Each row: pick a set of k+1 values with 0, increasing,
    whose diffs avoid `used`. Minimize the row's max (the scope contribution).
    Uses backtracking on a per-row basis with randomized restarts.
    """
    start = time.time()
    best_scope = None
    best_rows = None

    for attempt in range(200):
        if time.time() > deadline - 2.0 or time.time() > start + time_budget:
            break
        random.shuffle_order = None
        u = set(used)
        r = []
        ok = True
        for ri in range(n_rows):
            row, rowmax = _build_row_greedy(k, u, randomize=(attempt % 3 == 0))
            if row is None:
                ok = False
                break
            r.append(row)
            u |= _diffs_of_set(row)
        if ok:
            scope = max(max(x) for x in r)
            if best_scope is None or scope < best_scope:
                best_scope = scope
                best_rows = [list(x) for x in r]
                if best_scope == _counting_lower_bound(n_rows, k):
                    break  # optimal
    if best_rows is None:
        # fallback: pure deterministic greedy
        u = set(used)
        r = []
        for ri in range(n_rows):
            row, _ = _build_row_greedy(k, u, randomize=False)
            r.append(row)
            u |= _diffs_of_set(row)
        best_rows = r
        best_scope = max(max(x) for x in r)
    return best_rows, best_scope


def _counting_lower_bound(n_rows, k):
    total = n_rows * (k * (k + 1) // 2)
    return total  # scope >= total distinct diffs, each in [1..scope]


def _build_row_greedy(k, used, randomize=False):
    """
    Build one row (k+1 values, 0-inclusive, increasing) avoiding `used` diffs,
    minimizing the row's max value. Returns (row, maxval) or (None, None).
    """
    # Greedy: start [0]; repeatedly add the smallest candidate x > last such that
    # all new diffs {x - a : a in row} avoid used and are distinct within row.
    # To minimize max, we add the smallest feasible x each step.
    row = [0]
    used_local = set(used)
    x = 1
    guard = 0
    while len(row) < k + 1:
        guard += 1
        if guard > 500000:
            return None, None
        new_diffs = {x - a for a in row}
        if not (new_diffs & used_local) and len(new_diffs) == len(row):
            row.append(x)
            used_local |= new_diffs
            x = x + 1
        else:
            x += 1
    return row, row[-1]


def _scope_reduction_search(n_rows, k, init_rows, init_scope, deadline):
    """
    Structural search to reduce scope: iteratively attempt to repack with a
    smaller scope by treating row difference-sets as variables with a global
    disjointness constraint, solved by randomized constraint propagation +
    largest-difference-first assignment with backtracking.
    """
    best_rows = [list(r) for r in init_rows]
    best_scope = init_scope
    lb = _counting_lower_bound(n_rows, k)

    # We'll try scopes from best_scope-1 down toward lb.
    for target in range(best_scope - 1, lb - 1, -1):
        if time.time() > deadline - 3.0:
            break
        # Attempt to find a valid packing within scope `target`.
        found = _try_pack_scope(n_rows, k, target, deadline, max_time=8.0)
        if found is not None:
            best_rows = found
            best_scope = target
            # keep trying to go lower
            continue
        else:
            # can't fit in this scope (within budget); stop descending
            break

    return best_rows, best_scope


def _try_pack_scope(n_rows, k, scope, deadline, max_time):
    """
    Try to pack n_rows rows each with scope <= `scope`.
    Representation: each row = set of k+1 values in [0..scope] containing 0.
    Difference-sets must be disjoint. Solve via backtracking with
    most-constrained-first (largest available differences) and constraint
    propagation on the difference pool.

    Approach: think of the pool of available difference values [1..scope] minus
    used. Each row needs C(k+1,2) distinct values from the pool, AND those values
    must be realizable as a difference set of a 0-inclusive (k+1)-subset of [0..scope].

    We generate a pool of candidate rows (each a valid difference-set realizable
    within scope), then pick n_rows with disjoint difference-sets (a set-packing /
    matching problem) using backtracking ordered by largest difference.
    """
    start = time.time()
    # Generate candidate rows: all "greedy minimal" rows and randomized variants.
    candidates = []
    # deterministic greedy minimal rows (many will share small diffs; need variety)
    for variant in range(40):
        random.seed(1000 + variant * 7)
        u = set()
        row, mx = _build_row_in_scope(k, scope, randomize=True)
        if row is not None:
            candidates.append(row)
    # Also include the greedy-minimal (non-random) row
    row0, _ = _build_row_in_scope(k, scope, randomize=False)
    if row0 is not None:
        candidates.insert(0, row0)

    if len(candidates) < n_rows:
        return None

    # Precompute each candidate's diff set and max value.
    cand_info = []
    for row in candidates:
        d = _diffs_of_set(row)
        cand_info.append((row, d, max(d)))

    # Order candidates by largest difference descending (most constrained first
    # for the pool), but we need disjointness. Use backtracking set packing.
    cand_info.sort(key=lambda c: c[2], reverse=True)

    def solve(idx, used, chosen):
        nonlocal deadline
        if time.time() > deadline - 3.0 or time.time() > start + max_time:
            return None
        if len(chosen) == n_rows:
            return [c for c in chosen]
        # try candidates from idx
        for ci in range(idx, len(cand_info)):
            row, d, mx = cand_info[ci]
            if d & used:
                continue
            res = solve(ci + 1, used | d, chosen + [row])
            if res is not None:
                return res
        return None

    return solve(0, set(), [])


def _build_row_in_scope(k, scope, randomize=False):
    """
    Build a row of k+1 values in [0..scope], 0-inclusive, increasing, with all
    within-row differences distinct (Sidon-ish). If randomize, add jitter to
    candidate ordering for variety.
    """
    row = [0]
    used_local = set()
    x = 1
    guard = 0
    while len(row) < k + 1:
        guard += 1
        if guard > 200000:
            return None, None
        if x > scope:
            return None, None
        new_diffs = {x - a for a in row}
        if not (new_diffs & used_local) and len(new_diffs) == len(row):
            row.append(x)
            used_local |= new_diffs
            if randomize and random.random() < 0.3:
                x += random.randint(1, 3)
            else:
                x += 1
        else:
            x += 1
    return row, row[-1]


# EVOLVE-BLOCK-END
