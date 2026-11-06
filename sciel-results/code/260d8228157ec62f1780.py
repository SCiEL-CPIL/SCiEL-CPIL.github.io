import time
import random
import math
from math import gcd
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 145.0   # hard wall-clock guard (evaluator kills at 150s)

def _remaining():
    return _DEADLINE - time.time()

# ---------- exact score: max subset with no 3 collinear (bitmask B&B) ----------
def _max_indep_score(pts):
    n = len(pts)
    # bit i -> point i
    # triples: frozenset of 3 indices that are collinear
    # build collinearity: for each pair, count points on the line
    # Represent each triple as a 64-bit mask (n<=20)
    triple_masks = []
    # group points by line (using reduced direction + offset)
    line_pts = {}
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            g = gcd(abs(dx), abs(dy))
            if g:
                dx //= g; dy //= g
            # line key: normalized (a,b,c) for ax+by+c=0
            a = dy
            b = -dx
            c = -(a * xi + b * yi)
            gg = gcd(abs(a), gcd(abs(b), abs(c)))
            if gg:
                a //= gg; b //= gg; c //= gg
            # canonical sign
            if a < 0 or (a == 0 and (b < 0 or (b == 0 and c < 0))):
                a, b, c = -a, -b, -c
            line_pts.setdefault((a, b, c), set()).add(i)
            line_pts[(a, b, c)].add(j)
    # for each line with >=3 points, add all triples
    for key, idxs in line_pts.items():
        if len(idxs) >= 3:
            lst = list(idxs)
            for t in combinations(lst, 3):
                m = 0
                for v in t:
                    m |= (1 << v)
                triple_masks.append(m)
    # dedupe
    triple_masks = list(set(triple_masks))
    T = len(triple_masks)

    # precompute for each point the list of triple indices containing it
    pt_to_triples = [[] for _ in range(n)]
    for ti, m in enumerate(triple_masks):
        v = m
        while v:
            lsb = v & (-v)
            idx = lsb.bit_length() - 1
            pt_to_triples[idx].append(ti)
            v ^= lsb

    full = (1 << n) - 1
    best = [0]

    # A configuration is valid (no 3 collinear in selected set) iff it contains no triple.
    # We want max size of a set S with no triple fully inside S.
    # B&B over points: maintain current set mask, and a "candidate" mask of points still
    # available that do not, together with already-chosen points, form a forbidden triple.

    # For a chosen set C, a point p is still "safe" to add if for every triple containing p,
    # the other two are not both in C.
    def safe_mask(chosen):
        # start with all points not in chosen
        m = full ^ chosen
        # remove any point p such that exists triple {p,a,b} with a,b in chosen
        for p in range(n):
            if chosen & (1 << p):
                continue
            if m & (1 << p):
                for ti in pt_to_triples[p]:
                    others = triple_masks[ti] & ~(1 << p)
                    if others & chosen == others and others != 0:
                        # both others chosen -> p unsafe
                        m &= ~(1 << p)
                        break
        return m

    # ordering: sort points by degree (more triples first) to prune; but bitmask indices fixed.
    # We'll do a recursive search with a greedy upper bound.

    import sys
    sys.setrecursionlimit(10000)

    # Precompute a fast "upper bound": |chosen| + |safe_mask(chosen)|
    def search(chosen, safe, depth):
        sz = chosen.bit_count()
        if sz > best[0]:
            best[0] = sz
        if safe == 0:
            return
        # upper bound
        if sz + safe.bit_count() <= best[0]:
            return
        # pick a point to branch on: choose the one in safe that appears in most triples
        # (to prune). Use a heuristic: iterate safe points.
        # To keep it fast, branch on the first available point by a fixed order.
        # Choose point with max remaining-triple involvement.
        # Compute involvement: number of triples fully within (chosen ∪ safe) that contain p.
        # Approx: use static degree.
        # We'll branch on the point in safe with highest static degree.
        # Find it:
        cand = safe
        best_p = -1
        best_deg = -1
        # iterate
        v = cand
        # choose min index with max degree for determinism; but we want max degree.
        # compute degrees quickly
        while v:
            lsb = v & (-v)
            idx = lsb.bit_length() - 1
            d = len(pt_to_triples[idx])
            if d > best_deg:
                best_deg = d
                best_p = idx
            v ^= lsb
        p = best_p
        bit = 1 << p
        # Branch 1: include p
        new_chosen = chosen | bit
        new_safe = safe_mask(new_chosen) & (safe ^ bit)
        search(new_chosen, new_safe, depth + 1)
        # Branch 2: exclude p
        search(chosen, safe ^ bit, depth + 1)

    search(0, full, 0)
    return best[0]

# ---------- validity: no four collinear ----------
def _no_four_collinear(pts):
    n = len(pts)
    line_pts = {}
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            g = gcd(abs(dx), abs(dy))
            if g:
                dx //= g; dy //= g
            a = dy
            b = -dx
            c = -(a * xi + b * yi)
            gg = gcd(abs(a), gcd(abs(b), abs(c)))
            if gg:
                a //= gg; b //= gg; c //= gg
            if a < 0 or (a == 0 and (b < 0 or (b == 0 and c < 0))):
                a, b, c = -a, -b, -c
            line_pts.setdefault((a, b, c), set()).add(i)
            line_pts[(a, b, c)].add(j)
    for key, idxs in line_pts.items():
        if len(idxs) >= 4:
            return False
    return True

def _valid(pts):
    if len(set(tuple(p) for p in pts)) != len(pts):
        return False
    for x, y in pts:
        if abs(x) > 100 or abs(y) > 100:
            return False
    return _no_four_collinear(pts)

# ---------- candidate pool ----------
def _build_pool():
    pool = set()
    # parabola y = x^2 (within bounds)
    for x in range(-10, 11):
        y = x * x
        if abs(y) <= 100:
            pool.add((x, y))
    # semicircle x^2 + y^2 = 64, y>=0
    for x in range(-8, 9):
        yy = 64 - x * x
        if yy >= 0:
            y = int(round(math.sqrt(yy)))
            if x * x + y * y == 64 and y >= 0:
                pool.add((x, y))
    # circle x^2 + y^2 = 25
    for x in range(-5, 6):
        yy = 25 - x * x
        if yy >= 0:
            y = int(round(math.sqrt(yy)))
            if x * x + y * y == 25:
                pool.add((x, y))
    # hyperbola x*y = 12
    for x in range(-12, 13):
        if x == 0:
            continue
        if 12 % x == 0:
            y = 12 // x
            if abs(y) <= 100:
                pool.add((x, y))
    # line-ish convex: y = x + 30 (but that's collinear 3+ -> avoid many on one line)
    # Instead use a shallow parabola y = x//2 (integer) for variety
    for x in range(-20, 21):
        y = (x * x) // 4
        if abs(y) <= 100:
            pool.add((x, y))
    # another curve: x^2 - y^2 = 0 is degenerate; use y = x^2 - 20
    for x in range(-10, 11):
        y = x * x - 20
        if abs(y) <= 100:
            pool.add((x, y))
    # a few random points to break symmetry
    rnd = random.Random(12345)
    for _ in range(40):
        x = rnd.randint(-100, 100)
        y = rnd.randint(-100, 100)
        pool.add((x, y))
    pool = list(pool)
    return pool

def construct():
    pool = _build_pool()
    # initial configuration: pick 20 points that are valid (no 4 collinear)
    # Greedy: take points one by one from pool in a fixed order, keep if valid.
    # Use a deterministic-ish order: sort by (x, y).
    ordered = sorted(pool)
    cfg = []
    for p in ordered:
        cand = cfg + [p]
        if len(cand) == 20:
            if _valid(cand):
                cfg = cand
                break
        elif _valid(cand):
            cfg.append(p)
    if len(cfg) != 20:
        # fallback: parabola
        cfg = [[x, x * x] for x in range(-10, 10)]
        return cfg
    # convert to list of [x,y]
    cfg = [[p[0], p[1]] for p in cfg]
    best_score = _max_indep_score(cfg)
    best_cfg = cfg
    if _remaining() < 5:
        return best_cfg

    rnd = random.Random(777)
    T = 0.5
    T_min = 0.02
    # local search: swap one point for a pool point, or 2-swap
    pool_set = set(tuple(p) for p in pool)
    pool_list = list(pool)
    while _remaining() > 8:
        # choose move type
        if rnd.random() < 0.6:
            # 1-swap
            i = rnd.randrange(20)
            newp = rnd.choice(pool_list)
            if tuple(newp) == tuple(cfg[i]):
                continue
            newcfg = cfg[:]
            newcfg[i] = list(newp)
            # quick validity: distinct + no 4 collinear
            if not _valid(newcfg):
                continue
            ns = _max_indep_score(newcfg)
            if ns <= best_score or rnd.random() < math.exp((best_score - ns) / T):
                # accept
                if ns < best_score:
                    best_score = ns
                    best_cfg = newcfg
                cfg = newcfg
        else:
            # 2-swap
            i, j = rnd.sample(range(20), 2)
            newpi = rnd.choice(pool_list)
            newpj = rnd.choice(pool_list)
            newcfg = cfg[:]
            newcfg[i] = list(newpi)
            newcfg[j] = list(newpj)
            if not _valid(newcfg):
                continue
            ns = _max_indep_score(newcfg)
            if ns <= best_score or rnd.random() < math.exp((best_score - ns) / T):
                if ns < best_score:
                    best_score = ns
                    best_cfg = newcfg
                cfg = newcfg
        # cool
        T = max(T_min, T * 0.999)
    return best_cfg

# EVOLVE-BLOCK-START
# (entry point construct defined above)
# EVOLVE-BLOCK-END
