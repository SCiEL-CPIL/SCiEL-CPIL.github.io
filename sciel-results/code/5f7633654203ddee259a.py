"""Difference-triangle set via CSP over difference assignments + SA polish."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _scope(rows):
    return max(max(r) for r in rows)


def _row_diffs(row):
    """All positive pairwise differences of a row (k+1 choose 2 of them)."""
    out = []
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            out.append(row[j] - ri)
    return out


def _validate(rows):
    used = set()
    for r in rows:
        if r[0] != 0:
            return False
        for i in range(len(r)):
            if i > 0 and r[i] <= r[i - 1]:
                return False
        for d in _row_diffs(r):
            if d <= 0 or d in used:
                return False
            used.add(d)
    return True


def _greedy_fallback(n_rows, k):
    """Original greedy: always valid, used as safety net."""
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


def _fast_greedy(n_rows, k, rng):
    """Randomized greedy with a small lookahead window; returns a valid set."""
    used, rows = set(), []
    for r in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            window = 1 + (r % 5)
            cands = []
            for cand in range(x + 1, x + 1 + window):
                new = {cand - a for a in row}
                if not new & used:
                    cands.append((sum(new), cand))
            if cands:
                cands.sort()
                if rng.random() < 0.7:
                    chosen = cands[0][1]
                else:
                    chosen = cands[rng.randrange(len(cands))][1]
            else:
                x += 1
                chosen = x
            row.append(chosen)
            used |= {chosen - a for a in row[:-1]}
            x = chosen
        rows.append(row)
    return rows


def _diffs_from_entries(entries):
    """Given a list of (pos, value) sorted by pos, return the multiset of diffs."""
    out = []
    n = len(entries)
    for i in range(n):
        vi = entries[i][1]
        for j in range(i + 1, n):
            out.append(entries[j][1] - vi)
    return out


def _build_row_from_differences(diffs, k):
    """
    Given a row's k*(k+1)/2 differences (as a list, possibly with a placeholder
    -1 for the unassigned last difference), try to recover the row's entries.
    We know entries[0]=0 and entries are strictly increasing.
    The differences are over all pairs (i<j). We recover entries greedily:
      entries[0] = 0.
      For pos p (1..k), entries[p] must be such that (entries[p]-entries[i])
      for all i<p are exactly the differences that have 'p' as the larger index.
    We identify which differences belong to pos p: they are the ones that equal
    entries[p] - entries[i] for i<p. Since entries[p] is unknown, we try candidate
    values.
    This is complex; instead we do a direct DFS over entries using the diff set as
    a constraint (each new entry must produce only diffs already in the set, and
    consume them).
    """
    # diff multiset as a dict value->count
    from collections import Counter
    remaining = Counter()
    for d in diffs:
        if d > 0:
            remaining[d] += 1
    # DFS over entries
    n = k + 1
    entries = [0]
    sys.setrecursionlimit(10000)

    def dfs(pos):
        if pos == n:
            return entries[:]
        # candidate value v > entries[-1]
        start = entries[-1] + 1
        # The differences v - entries[i] for i in 0..pos-1 must all be in remaining
        # and distinct from each other (they are distinct since entries distinct).
        # We bound v: v - entries[0] = v must be in remaining.
        # Try v such that all (v - entries[i]) are in remaining.
        # Efficient: v must be in remaining (as v - 0). Iterate candidate v.
        for v in range(start, 2000):
            ok = True
            newdiffs = []
            for i in range(pos):
                d = v - entries[i]
                if remaining.get(d, 0) <= 0:
                    ok = False
                    break
                newdiffs.append(d)
            if not ok:
                continue
            # also ensure the newdiffs are mutually distinct (they are, since entries distinct)
            # consume
            for d in newdiffs:
                remaining[d] -= 1
            entries.append(v)
            res = dfs(pos + 1)
            if res is not None:
                return res
            entries.pop()
            for d in newdiffs:
                remaining[d] += 1
        return None

    return dfs(1)


def _csp_construct(n_rows, k, scope_cap, deadline, rng):
    """
    CSP over difference assignments.
    For each row we choose a set of k*(k+1)/2 distinct differences (all in 1..scope_cap),
    then recover the row's entries. All differences across all rows must be distinct.
    We build rows one at a time; for each row we enumerate candidate difference-sets
    (via backtracking with most-constrained heuristic) and pick the first that yields
    a valid row with max entry <= scope_cap and no diff collision with used.
    """
    total_diffs_per_row = k * (k + 1) // 2
    used = set()
    rows = []

    def build_one_row(used_so_far, deadline):
        """Return a valid row (list of entries) whose diffs avoid used_so_far and
        max entry <= scope_cap, or None."""
        # We need to pick total_diffs_per_row distinct values in 1..scope_cap,
        # none in used_so_far, that form a valid difference set of a row.
        # Backtrack: pick differences one at a time. The "most constrained" ordering:
        # we must pick differences such that they are realizable.
        # Realizability is global; so instead we directly DFS over entries bounded
        # by scope_cap, ensuring each produced diff is fresh.
        # This is a DFS over entries with pruning (forward checking on diffs).
        sys.setrecursionlimit(10000)

        def dfs_entries(pos, current):
            if time.time() > deadline:
                return None
            if pos == k + 1:
                return current[:]
            start = current[-1] + 1
            for v in range(start, scope_cap + 1):
                newdiffs = []
                ok = True
                for i in range(pos):
                    d = v - current[i]
                    if d in used_so_far:
                        ok = False
                        break
                    newdiffs.append(d)
                if not ok:
                    continue
                current.append(v)
                # forward check: reserve these diffs temporarily
                for d in newdiffs:
                    used_so_far.add(d)
                res = dfs_entries(pos + 1, current)
                for d in newdiffs:
                    used_so_far.discard(d)
                if res is not None:
                    return res
                current.pop()
            return None

        return dfs_entries(1, [0])

    for r in range(n_rows):
        if time.time() > deadline:
            return None
        row = build_one_row(used, deadline)
        if row is None:
            return None
        for d in _row_diffs(row):
            used.add(d)
        rows.append(row)
    return rows


def _sa_optimize(rows, k, deadline, rng):
    """Simulated annealing: perturb one entry of one row, accept if valid & improves
    or with prob exp(-dE/T). Minimize scope."""
    rows = [list(r) for r in rows]
    best = [list(r) for r in rows]
    best_sc = _scope(best)
    cur_sc = best_sc

    def all_diffs(rws):
        s = set()
        for r in rws:
            for d in _row_diffs(r):
                s.add(d)
        return s

    def row_valid(r):
        if r[0] != 0:
            return False
        for i in range(1, len(r)):
            if r[i] <= r[i - 1]:
                return False
        ds = _row_diffs(r)
        return len(ds) == len(set(ds))

    T = 20.0
    t0 = time.time()
    n = len(rows)
    while time.time() < deadline:
        # pick a row and a position to perturb
        ri = rng.randrange(n)
        row = rows[ri]
        pos = rng.randrange(1, len(row))
        old = row[pos]
        # candidate: move to a nearby value
        lo = row[pos - 1] + 1
        hi = old + 3 if old + 3 < scope_cap_global else old + 3
        # try to reduce scope: bias downward
        step = rng.randrange(-4, 5)
        cand = old + step
        if cand < lo or cand > row[pos + 1] - 1 if pos + 1 < len(row) else False:
            if pos + 1 < len(row):
                if cand < lo or cand >= row[pos + 1]:
                    continue
            else:
                if cand < lo:
                    continue
        if cand == old:
            continue
        row[pos] = cand
        if not row_valid(row):
            row[pos] = old
            continue
        # check diff collision with other rows
        rd = set(_row_diffs(row))
        conflict = False
        for rj in range(n):
            if rj == ri:
                continue
            for d in _row_diffs(rows[rj]):
                if d in rd:
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            row[pos] = old
            continue
        new_sc = _scope(rows)
        dE = new_sc - cur_sc
        if dE <= 0 or rng.random() < pow(2.718281828, -dE / max(T, 0.01)):
            cur_sc = new_sc
            if cur_sc < best_sc:
                best_sc = cur_sc
                best = [list(r) for r in rows]
        else:
            row[pos] = old
        # cool
        T = max(0.01, T * 0.999)
    return best, best_sc


scope_cap_global = 2000


def construct(n_rows, k):
    global scope_cap_global
    deadline = time.time() + 120.0  # safety margin before 150s kill
    rng = random.Random(777)

    # Phase 0: greedy fallback as baseline
    base_rows = _greedy_fallback(n_rows, k)
    best_rows = base_rows
    best_sc = _scope(base_rows)

    # Phase 1: randomized greedy restarts to get a tighter starting point
    t_phase1 = deadline - 60.0
    for _ in range(300000000):
        if time.time() > t_phase1:
            break
        rows = _fast_greedy(n_rows, k, rng)
        sc = _scope(rows)
        if sc < best_sc:
            best_sc = sc
            best_rows = rows

    # Phase 2: CSP with a tight scope cap to get a very tight initial set.
    # Start cap at current best and try to reduce.
    cap = best_sc
    while cap > 100 and time.time() < deadline - 45.0:
        scope_cap_global = cap
        rows = _csp_construct(n_rows, k, cap, deadline - 45.0, rng)
        if rows is not None and _validate(rows):
            sc = _scope(rows)
            if sc < best_sc:
                best_sc = sc
                best_rows = rows
            cap = max(100, sc - 3)
        else:
            # cap too tight or timed out; loosen a bit and try again
            cap = max(100, cap - 2)
            if cap >= best_sc:
                break

    # Phase 3: SA polish
    if time.time() < deadline - 5.0:
        scope_cap_global = max(best_sc + 50, 300)
        rows, sc = _sa_optimize(best_rows, k, deadline - 5.0, rng)
        if sc < best_sc:
            best_sc = sc
            best_rows = rows

    return best_rows

# EVOLVE-BLOCK-END
