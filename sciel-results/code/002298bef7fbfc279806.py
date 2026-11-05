"""
Difference triangle set (7,5) solver.

Structural change:
- Directed backtracking construction (rows built one at a time, not a max-clique
  over a random pool).
- Scope is minimized by binary search on the upper bound: for a given max entry
  M, try to build n_rows valid rows all with entries <= M.
- Variable ordering uses a "fragmentation + gap" heuristic (from cell findings):
  prefer placing a value that creates fewer isolated gaps in the used-difference
  set, plus a linear penalty for the value itself. This models the contiguous
  resource carving intuition and finds tighter packings.
- Repair + local-search polish machinery retained from the parent.
"""
import time
import random
import sys
import math

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


def _fragmentation_score(value, row, used):
    """
    Score a candidate next value: count how many NEW isolated gaps it creates
    in the used-difference set, plus a linear penalty on the value.
    Lower is better.
    """
    new_diffs = {value - a for a in row}
    # new_diffs is the set of differences this value adds
    # A "gap" is created when a difference d is added but neither d-1 nor d+1
    # is already in used (i.e., it starts a new isolated region).
    frag = 0
    for d in new_diffs:
        if d not in used:
            if (d - 1) not in used and (d + 1) not in used:
                frag += 1
    return frag * 10 + value


def _construct_row_directed(used, k, M, rng, deadline, heuristic_weight=1.0):
    """
    Build one row of k+1 increasing integers starting at 0, all <= M,
    avoiding used differences. Uses a fragmentation+gap heuristic to order
    candidates and backtracks if stuck.
    Returns (row, diffs) or None.
    """
    row = [0]
    new_diffs = set()
    used_local = set(used)  # we track within-row diffs separately

    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == k:  # need k more elements after 0, so row length k+1
            return True
        last = row[-1]
        # Candidate values: last+1 .. M
        # Order by fragmentation heuristic
        cands = []
        for v in range(last + 1, M + 1):
            nd = {v - a for a in row}
            if nd & used_local or nd & new_diffs:
                continue
            score = _fragmentation_score(v, row, used_local | new_diffs)
            cands.append((score, v))
        if not cands:
            return False
        cands.sort(key=lambda t: t[0])
        # Take a bounded number of top candidates to keep search tractable
        limit = min(len(cands), 12)
        # Occasionally shuffle the top few for diversity
        if rng.random() < 0.3 and limit > 1:
            top = cands[:limit]
            rng.shuffle(top[1:])  # keep best first, shuffle rest
            cands = top
        for _, v in cands[:limit]:
            row.append(v)
            nd = {v - a for a in row[:-1]}
            new_diffs |= nd
            if backtrack(pos + 1):
                return True
            row.pop()
            for a in row:
                new_diffs.discard(v - a)
        return False

    if not backtrack(0):
        return None, None
    return row, frozenset(new_diffs)


def _try_build_all(n_rows, k, M, rng, deadline):
    """
    Try to build n_rows valid rows, each with max entry <= M.
    Uses directed backtracking with heuristic ordering.
    Returns list of rows or None.
    """
    used = set()
    rows = []
    for r in range(n_rows):
        if time.time() > deadline:
            return None
        # Try multiple times with different random seeds / weights
        found = False
        for attempt in range(6):
            row, diffs = _construct_row_directed(
                used, k, M, rng, deadline,
                heuristic_weight=1.0 + attempt * 0.5
            )
            if row is not None:
                rows.append(row)
                used |= diffs
                found = True
                break
        if not found:
            return None
    return rows


def _local_search_scope(rows, k, rng, deadline):
    """
    Local search to reduce the scope (maximum entry).
    Tries to shrink elements in rows, with alternating orderings to break symmetry.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)

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

            # Try to reduce the last element
            last = row[-1]
            for new_last in range(last - 1, row[-2], -1):
                old_diffs = {last - a for a in row[:-1]}
                new_diffs = {new_last - a for a in row[:-1]}
                remaining = used - old_diffs
                if not (new_diffs & remaining):
                    row[-1] = new_last
                    used = (used - old_diffs) | new_diffs
                    improved = True
                    break

            # Try to reduce other elements
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
                        improved = True
                        break

    return rows


def _polish_pass(rows, k, deadline):
    """
    High-precision polish: try every possible reduction for each position.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    best_scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]

    for _ in range(10):
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

        current_scope = _total_scope(rows)
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [r[:] for r in rows]

    return best_rows


def _repair_pass(rows, k, rng, deadline):
    """
    Try to replace a row with a smaller-scope alternative that frees differences
    for other rows to shrink.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    scope = _total_scope(rows)
    best_rows = [r[:] for r in rows]
    best_scope = scope

    n = len(rows)
    for _ in range(40):
        if time.time() > deadline:
            break
        idx = rng.randrange(n)
        other_rows = [r for j, r in enumerate(rows) if j != idx]
        other_used = set()
        for r in other_rows:
            other_used |= _diffs_of_row(r)

        # Try to build a new row with scope <= best_scope
        new_row, new_diffs = _construct_row_directed(
            other_used, k, best_scope - 1, rng, deadline
        )
        if new_row is None:
            continue
        trial = [r[:] for r in rows]
        trial[idx] = new_row
        trial_scope = _total_scope(trial)
        if trial_scope < best_scope:
            best_rows = trial
            best_scope = trial_scope

    return best_rows if best_scope < scope else rows


def _greedy_construct(n_rows, k):
    """Pure greedy construction as fallback."""
    used = set()
    rows = []
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


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # --- Phase 1: Get a valid baseline via pure greedy ---
    rows = _greedy_construct(n_rows, k)
    best_rows = [r[:] for r in rows]
    best_scope = _total_scope(rows)

    # --- Phase 2: Binary search on scope using directed backtracking ---
    # Lower bound: counting bound ~ k*(k+1)/2 * n_rows / (k+1) = k*n_rows/2
    lo = max(k * n_rows // 2, 50)
    hi = best_scope

    # First, try to beat the greedy baseline by tightening from hi downward
    # Use binary search: find smallest M for which we can build all rows.
    # But binary search may find an M that's not locally optimal, so we also
    # do a linear descent near the optimum.
    search_deadline = deadline - 30  # leave time for polish

    # Binary search for the smallest feasible M
    feasible_M = None
    for _ in range(12):  # cap iterations
        if time.time() > search_deadline:
            break
        mid = (lo + hi) // 2
        if mid <= lo:
            break
        result = _try_build_all(n_rows, k, mid, rng, search_deadline)
        if result is not None:
            feasible_M = mid
            hi = mid
            # Polish this result immediately
            polished = _local_search_scope(result, k, rng, search_deadline)
            ps = _total_scope(polished)
            if ps < best_scope:
                best_scope = ps
                best_rows = [r[:] for r in polished]
            lo = max(lo, mid - 5)  # keep searching slightly lower
        else:
            lo = mid + 1

    # --- Phase 3: Linear descent from current best ---
    # Try to build with progressively smaller scope
    current_best = best_scope
    for M in range(current_best - 1, max(current_best - 30, lo - 1), -1):
        if time.time() > search_deadline:
            break
        result = _try_build_all(n_rows, k, M, rng, search_deadline)
        if result is not None:
            polished = _local_search_scope(result, k, rng, search_deadline)
            ps = _total_scope(polished)
            if ps < best_scope:
                best_scope = ps
                best_rows = [r[:] for r in polished]
                current_best = ps
        else:
            # Can't build at M, try M-1 (maybe a different construction works)
            continue

    # --- Phase 4: Polish the best result found ---
    if time.time() < deadline - 10:
        best_rows = _local_search_scope(best_rows, k, rng, deadline - 10)
        best_scope = _total_scope(best_rows)

    # --- Phase 5: Repair pass to escape local minima ---
    if time.time() < deadline - 8:
        repaired = _repair_pass(best_rows, k, rng, deadline - 8)
        rs = _total_scope(repaired)
        if rs < best_scope:
            best_scope = rs
            best_rows = repaired

    # --- Phase 6: High-precision polish ---
    if time.time() < deadline - 3:
        best_rows = _polish_pass(best_rows, k, deadline - 3)
        best_scope = _total_scope(best_rows)

    # --- Phase 7: Final local search ---
    if time.time() < deadline - 1:
        best_rows = _local_search_scope(best_rows, k, rng, deadline - 1)

    return best_rows

# EVOLVE-BLOCK-END
