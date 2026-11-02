"""Modular-lattice construction for sumset/difference-set exponent lower bound."""
# EVOLVE-BLOCK-START
import math
import time
import random
import itertools

_DEADLINE = time.time() + 110.0  # safety margin before 150s wall-clock kill

def _score_set(U):
    """Lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U:
        return 0.0
    s = set(U)
    if len(s) < 2:
        return 0.0
    m = max(U)
    if m <= 0:
        return 0.0
    # sumset and difference set
    su = set()
    du = set()
    lst = list(s)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for b in lst[i:]:
            su.add(a + b)
            du.add(a - b)
            if a != b:
                du.add(b - a)
    # include 0 in difference
    du.add(0)
    lu = len(su)
    ld = len(du)
    if ld <= 0 or lu <= 0:
        return 0.0
    denom = math.log(2 * m + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ld / lu) / denom

def _build_from_mod(m, r, seed=0):
    """Build U as a dense sublattice in Z_m: multiples of m/r up to some max,
    plus 0. r is the density (number of residue classes step)."""
    if m < 2 or r < 1:
        return [0]
    # elements are multiples of m/r within [0, m) plus possibly up to 2m to enlarge spread
    step = m // r
    if step < 1:
        step = 1
    # choose a max spread to keep difference set large: use up to ~2m
    maxv = 2 * m
    U = set()
    x = 0
    while x <= maxv:
        U.add(x)
        x += step
    U.add(0)
    return sorted(U)

def _local_search(U, budget_time, seed=0):
    """Refine U by adding/removing/adjusting elements to improve score."""
    rng = random.Random(seed)
    U = sorted(set(U))
    best = _score_set(U)
    bestU = list(U)
    start = time.time()
    max_elems = max(60, len(U) * 2)
    while time.time() - start < budget_time and len(U) < max_elems:
        improved = False
        m = max(U) if U else 0
        # candidate additions: small gaps and structured points
        cands = set()
        # fill small gaps near 0
        for g in range(1, max(5, m // 20 + 1)):
            if g not in U:
                cands.add(g)
        # structured: powers and multiples
        for p in [2, 3, 5, 7, 11]:
            v = p
            while v <= 2 * m + 10:
                cands.add(v)
                v *= p
        # random candidates
        for _ in range(30):
            cands.add(rng.randint(0, 2 * m + 20))
        # try each candidate addition
        for c in sorted(cands):
            if c in U:
                continue
            newU = U + [c]
            sc = _score_set(newU)
            if sc > best + 1e-12:
                best = sc
                bestU = newU
                U = sorted(newU)
                improved = True
        # try removals (keep 0)
        for c in list(U):
            if c == 0:
                continue
            newU = [x for x in U if x != c]
            if len(newU) < 2:
                continue
            sc = _score_set(newU)
            if sc > best + 1e-12:
                best = sc
                bestU = newU
                U = sorted(newU)
                improved = True
        if not improved:
            break
    return bestU, best

def construct():
    start = time.time()
    rng = random.Random(12345)
    bestU = [0, 1, 3]
    best = _score_set(bestU)

    # Phase 1: greedy modular-lattice search over (m, r)
    # Try a range of moduli and densities
    m_list = []
    # small to moderate moduli
    for m in range(4, 400, 3):
        m_list.append(m)
    # some larger
    for m in range(400, 2000, 37):
        m_list.append(m)
    # spread
    for m in range(2000, 60000, 997):
        m_list.append(m)

    for m in m_list:
        if time.time() > _DEADLINE - 20:
            break
        # density r: number of residue classes (step = m/r)
        r_options = [1, 2, 3, 4, 5, 8, 10, 16, 20, 32, 50, 64, 100]
        r_options = [r for r in r_options if r <= m]
        for r in r_options:
            if time.time() > _DEADLINE - 20:
                break
            U = _build_from_mod(m, r)
            if len(U) < 2:
                continue
            sc = _score_set(U)
            if sc > best + 1e-12:
                best = sc
                bestU = U

    # Phase 2: local search refinement on the best found
    if time.time() < _DEADLINE - 10:
        budget = _DEADLINE - 10 - time.time()
        refined, rsc = _local_search(bestU, budget, seed=99)
        if rsc > best + 1e-12:
            best = rsc
            bestU = refined

    # Ensure 0 present and nonnegative
    bestU = sorted(set(x for x in bestU if x >= 0))
    if 0 not in bestU:
        bestU = [0] + bestU
    if len(bestU) < 2:
        bestU = [0, 1, 3]
    return bestU
# EVOLVE-BLOCK-END
