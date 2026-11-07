"""Fresh design: smooth-surrogate guided greedy + fast feasibility oracle + polish."""
import time
import math
from itertools import combinations

N = 128
UNIVERSE = list(range(1, N + 1))
DEADLINE = 88.0          # wall-clock safety margin before the 90s kill
START = time.time()

# Precompute all bad triples (a<b<c) and their pair sums (x=a+b, y=a+c, z=b+c).
TRIPLES = []
for a, b, c in combinations(UNIVERSE, 3):
    TRIPLES.append((a, b, c, a + b, a + c, b + c))
# Precompute, for each value v, the list of (a,b,c) indices where v is one of the pair sums.
SUMS_IN = {}
for (a, b, c, x, y, z) in TRIPLES:
    for v in (x, y, z):
        SUMS_IN.setdefault(v, []).append((a, b, c))

def _elapsed():
    return time.time() - START

def _bad_count(sel):
    """Number of bad triples fully present in sel."""
    cnt = 0
    for (a, b, c, x, y, z) in TRIPLES:
        if a in sel and b in sel and c in sel and x in sel and y in sel and z in sel:
            cnt += 1
    return cnt

def _logsumexp(terms):
    if not terms:
        return 0.0
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))

def _soft_penalty(sel):
    """Smooth surrogate: log(1 + sum exp(k * t_bad)) over bad triples."""
    k = 1.2
    terms = []
    for (a, b, c, x, y, z) in TRIPLES:
        if a in sel and b in sel and c in sel:
            present = (x in sel) + (y in sel) + (z in sel)
            if present >= 3:
                terms.append(k * (present - 2.0))  # >= k*1 when all three present
    if not terms:
        return 0.0
    return _logsumexp(terms)

def _objective(sel):
    return len(sel) - _soft_penalty(sel)

def _greedy_build(order):
    """Greedy add in given order; accept a vertex only if it keeps the set bad-triple-free."""
    sel = set()
    for v in order:
        # Tentatively add v; check if any bad triple becomes fully present.
        ok = True
        for (a, b, c) in SUMS_IN.get(v, ()):
            if a in sel and b in sel and c in sel:
                if a + b in sel and a + c in sel and b + c in sel:
                    ok = False
                    break
        if ok:
            sel.add(v)
    return sel

def _polish(sel, budget=2.0):
    """Try adding any remaining vertex that preserves feasibility; repeat until stable."""
    sel = set(sel)
    changed = True
    while changed and (_elapsed() - budget) < 0:
        changed = False
        for v in UNIVERSE:
            if v in sel:
                continue
            ok = True
            for (a, b, c) in SUMS_IN.get(v, ()):
                if a in sel and b in sel and c in sel:
                    if a + b in sel and a + c in sel and b + c in sel:
                        ok = False
                        break
            if ok:
                sel.add(v)
                changed = True
    return sel

def _local_search(sel, budget=4.0):
    """Swap-based local search: replace a selected vertex with an unselected one if it
    increases size while staying feasible, or enables a net gain. Simple hill-climb."""
    sel = set(sel)
    best = len(sel)
    improved = True
    while improved and (_elapsed() - budget) < 0:
        improved = False
        for out in list(sel):
            for inn in UNIVERSE:
                if inn in sel:
                    continue
                cand = (sel - {out}) | {inn}
                # feasibility check
                bad = False
                for (a, b, c, x, y, z) in TRIPLES:
                    if a in cand and b in cand and c in cand and x in cand and y in cand and z in cand:
                        bad = True
                        break
                if not bad and len(cand) > best:
                    sel = cand
                    best = len(cand)
                    improved = True
                    break
            if improved:
                break
    return sel

def _validate(sel):
    """Return True iff no bad triple is fully present."""
    for (a, b, c, x, y, z) in TRIPLES:
        if a in sel and b in sel and c in sel and x in sel and y in sel and z in sel:
            return False
    return True

def construct():
    deadline = _elapsed() + (DEADLINE - _elapsed())
    deadline = min(deadline, 88.0)

    # Deterministic baseline: upper half (the seed).
    best = set(range(N // 2 + 1, N + 1))

    # Greedy over several vertex orderings guided by the smooth surrogate.
    orderings = [
        sorted(UNIVERSE, reverse=True),
        sorted(UNIVERSE, key=lambda v: -len(SUMS_IN.get(v, ()))),
        sorted(UNIVERSE, key=lambda v: (v % 2, -v)),
        list(reversed(UNIVERSE)),
    ]
    for order in orderings:
        if _elapsed() > deadline:
            break
        cand = _greedy_build(order)
        cand = _polish(cand, budget=1.0)
        if _validate(cand) and len(cand) > len(best):
            best = cand

    # Polish the best.
    best = _polish(best, budget=1.5)

    # Local search with remaining time.
    if _elapsed() < deadline:
        cand = _local_search(best, budget=deadline - _elapsed() - 1.0)
        if _validate(cand) and len(cand) > len(best):
            best = cand

    # Final safety polish.
    best = _polish(best, budget=0.5)

    # Guarantee validity; if somehow invalid, fall back to a known-good subset.
    if not _validate(best):
        # Remove vertices to break all bad triples (crude but safe).
        for (a, b, c, x, y, z) in TRIPLES:
            if a in best and b in best and c in best and x in best and y in best and z in best:
                best.discard(z)
        best = _polish(best, budget=0.5)

    result = [int(v) for v in sorted(best)]
    return result
