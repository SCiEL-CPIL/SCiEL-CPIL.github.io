"""Construct up to 48 distinct ints in [1,192] no element divides sum of 3 distinct others."""
import time

N = 192
K = 3
MAXC = 48
DEADLINE = 80.0  # wall-clock safety margin under 90s kill


def _three_sum_table(selected):
    """Return boolean list where table[s] is True if s is sum of 3 distinct elements of selected."""
    n = len(selected)
    # all pairs sums
    pair = {}
    for i in range(n):
        ai = selected[i]
        for j in range(i + 1, n):
            pair[ai + selected[j]] = pair.get(ai + selected[j], 0) + 1
    # For 3 distinct: sum = a+b+c. We can iterate pairs and add a third distinct element.
    # Efficient: table size up to 3*192=576.
    table = [False] * (3 * N + 1)
    for i in range(n):
        ai = selected[i]
        for j in range(i + 1, n):
            s2 = ai + selected[j]
            for k in range(j + 1, n):
                s = s2 + selected[k]
                if s <= 3 * N:
                    table[s] = True
    return table


def _candidate_ok(x, selected):
    """Check if adding x keeps property: no element a in (selected + [x]) divides sum of 3 distinct others in that set."""
    newset = selected + [x]
    n = len(newset)
    # 1) x must not divide any 3-sum of distinct elements from `selected` (others)
    # 2) each a in selected must not divide any 3-sum of distinct elements from newset \ {a}
    # Build 3-sum table for newset once (covers all triples of distinct elements).
    table = [False] * (3 * N + 1)
    for i in range(n):
        ai = newset[i]
        for j in range(i + 1, n):
            s2 = ai + newset[j]
            for k in range(j + 1, n):
                s = s2 + newset[k]
                if s <= 3 * N:
                    table[s] = True
    # check x
    if x <= 3 * N:
        # x divides some triple sum of others (triple from selected only, since x not in triple)
        # Need triple of 3 distinct from selected. Build separate check.
        sel = selected
        m = len(sel)
        bad = False
        for i in range(m):
            si = sel[i]
            for j in range(i + 1, m):
                s2 = si + sel[j]
                for k in range(j + 1, m):
                    s = s2 + sel[k]
                    if s % x == 0:
                        bad = True
                        break
                if bad:
                    break
            if bad:
                break
        if bad:
            return False
    # check each a in selected: a must not divide any triple of 3 distinct from newset \ {a}
    for idx, a in enumerate(selected):
        # triples from newset excluding index idx
        others = [newset[t] for t in range(n) if t != idx]
        m = len(others)
        bad = False
        for i in range(m):
            si = others[i]
            for j in range(i + 1, m):
                s2 = si + others[j]
                for k in range(j + 1, m):
                    s = s2 + others[k]
                    if s % a == 0:
                        bad = True
                        break
                if bad:
                    break
            if bad:
                break
        if bad:
            return False
    return True


def _is_valid_full(selected):
    """Full validation of a set (independent of construction order)."""
    n = len(selected)
    for idx, a in enumerate(selected):
        others = [selected[t] for t in range(n) if t != idx]
        m = len(others)
        for i in range(m):
            si = others[i]
            for j in range(i + 1, m):
                s2 = si + others[j]
                for k in range(j + 1, m):
                    s = s2 + others[k]
                    if s % a == 0:
                        return False
    return True


def _greedy(order, start_set=None, deadline=None):
    selected = list(start_set) if start_set else []
    # precompute which are already in
    present = set(selected)
    for x in order:
        if x in present:
            continue
        if deadline is not None and time.time() > deadline:
            break
        if _candidate_ok(x, selected):
            selected.append(x)
            present.add(x)
    return selected


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE

    # Strategy: large numbers first (safe), then small.
    # Try several orderings to maximize cardinality.
    best = []

    # Base: all numbers 100..192 are very likely mutually safe. Verify quickly.
    base_candidates = list(range(100, N + 1))
    base = []
    for x in base_candidates:
        if _candidate_ok(x, base):
            base.append(x)

    # Now fill with small numbers in various orders.
    small = list(range(1, 100))
    orders = [
        small[::-1],   # descending small
        small,          # ascending small
        sorted(small, key=lambda v: -v),
    ]
    # Add a few heuristic orders: prefer numbers with fewer divisors relationships
    for order in orders:
        if time.time() > deadline:
            break
        res = _greedy(order, start_set=list(base), deadline=deadline)
        if len(res) > len(best):
            best = res

    # Also try starting from scratch with different orders (no base) in case base hurts.
    for order in [list(range(N, 0, -1)), list(range(1, N + 1))]:
        if time.time() > deadline:
            break
        res = _greedy(order, start_set=None, deadline=deadline)
        if len(res) > len(best):
            best = res

    # Local improvement: try to add any remaining number not in best.
    changed = True
    while changed and time.time() < deadline:
        changed = False
        present = set(best)
        for x in range(1, N + 1):
            if x in present:
                continue
            if _candidate_ok(x, best):
                best.append(x)
                present.add(x)
                changed = True

    # Final safety: verify validity; if invalid, fall back to a known-good small set.
    if not _is_valid_full(best):
        # Fallback: return large numbers only (very safe)
        fb = []
        for x in range(N, 0, -1):
            if _candidate_ok(x, fb):
                fb.append(x)
        if _is_valid_full(fb):
            best = fb
        else:
            # last resort: just large range
            best = [x for x in range(120, N + 1)]
            if not _is_valid_full(best):
                best = list(range(150, N + 1))

    return best
