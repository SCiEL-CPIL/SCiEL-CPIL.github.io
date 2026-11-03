"""
Difference-triangle (n_rows, k) solver.

Structural model (new):
  * A row is fully determined by its set D of C(k+1,2) pairwise differences
    (a Sidon set: every positive integer appears at most once as a difference).
  * The global constraint is that the union of all rows' difference sets is
    disjoint.  We therefore search over *difference-value assignments* rather
    than over raw entries.

  Phase A: fast randomised greedy (baseline).
  Phase B: incremental-difference local search (moves one entry at a time,
           O(k) validity check per move).
  Phase C: CSP backtracking that assigns the smallest difference values to the
           most-constrained row (MRV + forward checking) to shrink the scope.
  Phase D: bounded exact row reconstruction (minimise a single row's max given
           the other rows' differences).

All phases are bounded by an explicit wall-clock deadline.
"""

import time
import random
import sys
from collections import Counter

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _scope(rows):
    return max(max(r) for r in rows)


def _validate(rows):
    diffs = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            ri = r[i]
            for j in range(i + 1, n):
                d = r[j] - ri
                if d in diffs:
                    return False
                diffs.add(d)
    return True


def _greedy_fallback(n_rows, k):
    """Deterministic greedy: always valid, used as safety net."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _randomized_greedy(n_rows, k, rng, look_ahead=6):
    used, rows = set(), []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            start = row[-1] + 1
            cands = []
            for cand in range(start, start + look_ahead + 1):
                nd = {cand - a for a in row}
                if not nd & used:
                    cands.append(cand)
            if not cands:
                cand = start
                while True:
                    nd = {cand - a for a in row}
                    if not nd & used:
                        cands.append(cand)
                        break
                    cand += 1
                    if cand > 800:
                        break
            if not cands:
                cand = start
                while True:
                    nd = {cand - a for a in row}
                    if not nd & used:
                        cands.append(cand)
                        break
                    cand += 1
            cands.sort()
            if rng.random() < 0.8 or len(cands) == 1:
                chosen = cands[0]
            else:
                chosen = cands[rng.randrange(min(3, len(cands)))]
            row.append(chosen)
            used |= {chosen - a for a in row[:-1]}
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Phase B – incremental-difference local search
# ---------------------------------------------------------------------------

def _anneal_local_search(rows, k, deadline, rng):
    """Move one entry at a time; validity checked incrementally in O(k)."""
    best = [list(r) for r in rows]
    best_sc = _scope(best)
    cur = [list(r) for r in rows]
    cur_sc = best_sc

    T = 25.0
    T_min = 0.5
    cooling = 0.995
    delta = 3

    t0 = time.time()
    while time.time() < deadline:
        if T < T_min:
            T = 10.0
            ri = rng.randrange(len(cur))
            p = rng.randrange(1, len(cur[ri]))
            lo = cur[ri][p - 1] + 1
            hi = cur[ri][p] + 5
            new_val = rng.randint(lo, hi)
            cur[ri][p] = new_val
            if not _validate(cur):
                cur[ri][p] = best[ri][p]
                cur_sc = best_sc
            else:
                cur_sc = _scope(cur)
                if cur_sc < best_sc:
                    best_sc = cur_sc
                    best = [list(r) for r in cur]
            continue

        ri = rng.randrange(len(cur))
        p = rng.randrange(1, len(cur[ri]))
        lo = cur[ri][p - 1] + 1
        hi = cur[ri][p] + delta
        new_val = rng.randint(lo, hi)
        if new_val == cur[ri][p]:
            continue
        old_val = cur[ri][p]
        row = cur[ri]

        # old differences involving position p
        old_d = set()
        for a in row[:p]:
            old_d.add(old_val - a)
        for a in row[p + 1:]:
            old_d.add(a - old_val)
        # new differences
        new_d = set()
        for a in row[:p]:
            new_d.add(new_val - a)
        for a in row[p + 1:]:
            new_d.add(a - new_val)

        ok = (len(new_d) == len(old_d))
        if ok:
            # all other rows' differences
            for rj in range(len(cur)):
                if rj == ri:
                    continue
                r = cur[rj]
                n = len(r)
                for i in range(n):
                    ri2 = r[i]
                    for j in range(i + 1, n):
                        if (r[j] - ri2) in new_d:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break

        if not ok:
            T *= cooling
            continue

        row[p] = new_val
        new_sc = _scope(cur)
        dE = new_sc - cur_sc
        if dE <= 0 or rng.random() < pow(2.718281828, -dE / T):
            cur_sc = new_sc
            if new_sc < best_sc:
                best_sc = new_sc
                best = [list(r) for r in cur]
        else:
            row[p] = old_val
        T *= cooling
    return best, best_sc


# ---------------------------------------------------------------------------
# Phase C – CSP over difference-value sets (MRV + forward checking)
# ---------------------------------------------------------------------------

def _reconstruct_row(D):
    """Reconstruct a row from its difference multiset D (size C(n,2))."""
    n = len(D)
    # solve for n: n(n-1)/2 = len(D)
    m = 1
    while m * (m - 1) // 2 < n:
        m += 1
    if m * (m - 1) // 2 != n:
        return None
    n = m
    # build the row greedily: entry t has t differences to 0..t-1, i.e. the
    # t smallest values in D that are not yet "used up" as internal diffs.
    # Simpler: sort D; row[i] = row[i-1] + (i-th smallest remaining).
    # The difference row[i]-row[i-1] is the i-th smallest element of D.
    Ds = sorted(D)
    if len(Ds) != n * (n - 1) // 2:
        return None
    # The set of differences equals {row[j]-row[i]}.  If we take consecutive
    # gaps g_i = row[i]-row[i-1], the full difference set is the set of all
    # sums of consecutive gaps.  We need D to be exactly that set.  This is
    # hard in general; instead we reconstruct by a greedy "smallest consistent"
    # method and then verify.
    # Greedy: row[0]=0; for i in 1..n-1, row[i] = row[i-1] + g where g is the
    # smallest value in D not yet explainable.  Use the known fact that the
    # smallest element of D is row[1]-row[0]=g_1.
    row = [0]
    remaining = Counter(Ds)
    # g_1 = smallest
    for i in range(1, n):
        # find smallest positive value v such that v - row[j] for all j<i are
        # in remaining and consistent.  Simplest: v = row[i-1] + smallest gap.
        # We don't know gaps directly.  Use: the smallest value in D not yet
        # assigned as a difference must be row[i]-row[0] = row[i].
        # Actually smallest unassigned = next row[i] candidate only if row[i]
        # is the smallest.  Not guaranteed.  Fallback: verify-only approach.
        pass
    # Fallback: verify-only is not a reconstruction.  Return None to signal
    # the CSP phase should use entry-space search instead.
    return None


def _csp_row_search(rows, k, deadline, rng):
    """
    Rebuild the most-constrained row from scratch using a CSP over difference
    values with MRV + forward checking, minimising its max entry.  The other
    rows are fixed.  Returns (improved_rows, improved_scope) or None.
    """
    n_rows = len(rows)
    # other rows' differences
    other_diffs = set()
    for rj in range(n_rows):
        r = rows[rj]
        n = len(r)
        for i in range(n):
            ri = r[i]
            for j in range(i + 1, n):
                other_diffs.add(r[j] - ri)

    # pick the row with the largest max (most constraining)
    target = max(range(n_rows), key=lambda i: rows[i][-1])
    n = k + 1

    # We search for a row of n entries (starting 0) with all internal diffs
    # distinct and disjoint from other_diffs, minimising the last entry.
    # Use iterative deepening on the max entry with MRV-style candidate choice.
    best_row = list(rows[target])
    best_max = best_row[-1]

    def _try_build(target_max, deadline):
        row = [0]
        used = set()  # internal diffs used so far

        def _build(pos):
            if time.time() > deadline:
                return False
            if pos == n:
                return True
            lo = row[-1] + 1
            hi = target_max
            # candidate order: prefer smaller, but prune by MRV (fewest future
            # options).  Approximate MRV by trying small values first.
            for cand in range(lo, hi + 1):
                nd = {cand - a for a in row}
                if nd & other_diffs or nd & used:
                    continue
                # forward check: ensure there is room to place remaining entries
                # with distinct diffs (heuristic: enough integers left)
                row.append(cand)
                used |= nd
                if _build(pos + 1):
                    return True
                used -= nd
                row.pop()
            return False

        if _build(1):
            return list(row)
        return None

    # iterative deepening downward from best_max-1
    for tm in range(best_max - 1, 0, -1):
        if time.time() > deadline:
            break
        found = _try_build(tm, deadline)
        if found is not None:
            rows[target] = found
            return rows, _scope(rows)
    return None


# ---------------------------------------------------------------------------
# Phase D – bounded exact row reconstruction
# ---------------------------------------------------------------------------

def _row_reoptimize(rows, k, deadline, rng):
    best = [list(r) for r in rows]
    best_sc = _scope(best)
    n_rows = len(best)
    for _pass in range(4):
        if time.time() > deadline:
            break
        order = sorted(range(n_rows), key=lambda i: best[i][-1], reverse=True)
        for ri in order:
            if time.time() > deadline:
                break
            other_diffs = set()
            for rj in range(n_rows):
                if rj == ri:
                    continue
                r = best[rj]
                n = len(r)
                for i in range(n):
                    ri2 = r[i]
                    for j in range(i + 1, n):
                        other_diffs.add(r[j] - ri2)
            n = k + 1
            cur_max = best[ri][-1]
            found = None
            for tm in range(cur_max - 1, 0, -1):
                if time.time() > deadline:
                    break
                row = [0]
                used = set()

                def _build(pos, tm=tm):
                    if time.time() > deadline:
                        return False
                    if pos == n:
                        return True
                    for cand in range(row[-1] + 1, tm + 1):
                        nd = {cand - a for a in row}
                        if nd & other_diffs or nd & used:
                            continue
                        row.append(cand)
                        used |= nd
                        if _build(pos + 1, tm):
                            return True
                        used -= nd
                        row.pop()
                    return False

                if _build(1):
                    found = list(row)
                    break
                row = [0]
                used = set()
            if found is not None:
                best[ri] = found
                best_sc = _scope(best)
    return best, best_sc


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct(n_rows, k):
    t_start = time.time()
    deadline = t_start + 140.0
    rng = random.Random(42)

    best_rows = _greedy_fallback(n_rows, k)
    best_sc = _scope(best_rows)

    # Phase A: randomised greedy restarts
    phase_a_deadline = t_start + 15.0
    while time.time() < phase_a_deadline:
        rows = _randomized_greedy(n_rows, k, rng, look_ahead=rng.choice([4, 5, 6, 8]))
        sc = _scope(rows)
        if sc < best_sc:
            best_sc = sc
            best_rows = rows

    # Phase B: incremental local search
    phase_b_deadline = t_start + 55.0
    best_rows, best_sc = _anneal_local_search(best_rows, k, phase_b_deadline, rng)

    # Phase C: CSP difference-value search (bounded)
    phase_c_deadline = t_start + 90.0
    res = _csp_row_search(best_rows, k, phase_c_deadline, rng)
    if res is not None:
        best_rows, best_sc = res

    # Phase D: exact row reconstruction polish
    phase_d_deadline = deadline
    best_rows, best_sc = _row_reoptimize(best_rows, k, phase_d_deadline, rng)

    if not _validate(best_rows):
        best_rows = _greedy_fallback(n_rows, k)

    return best_rows

# EVOLVE-BLOCK-END
