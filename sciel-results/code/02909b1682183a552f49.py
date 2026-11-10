"""Geometric hypergraph design: 20 pts, no 4 collinear, minimize no-3-collinear subset."""
import time
import random

_DEADLINE = 110.0  # wall-clock safety margin before the 150s kill
_START = None
_RNG = None

def _now():
    return time.time()

def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]

def _has_four_collinear(pts):
    n = len(pts)
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[j][0] - pts[i][0]
            dy = pts[j][1] - pts[i][1]
            g = abs(dx)
            while g:
                g, dx = dx % g, g
            g = abs(dy)
            while g:
                g, dy = dy % g, g
            if g == 0:
                g = 1
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            key = (pts[i][0], pts[i][1], dx, dy)
            if key in seen:
                return True
            seen.add(key)
    return False

def _collinear_triples(pts):
    """Return list of (i,j,k) with i<j<k collinear."""
    n = len(pts)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[j][0] - pts[i][0]
            dy = pts[j][1] - pts[i][1]
            for k in range(j + 1, n):
                if _cross((dx, dy), (pts[k][0] - pts[i][0], pts[k][1] - pts[i][1])) == 0:
                    triples.append((i, j, k))
    return triples

def _build_B(pts):
    n = len(pts)
    B = [0] * n
    for (i, j, k) in _collinear_triples(pts):
        B[i] |= (1 << j) | (1 << k)
        B[j] |= (1 << i) | (1 << k)
        B[k] |= (1 << i) | (1 << j)
    return B

def _max_indep(B, n):
    """Exact maximum subset size with no three collinear, via branch and bound."""
    allmask = (1 << n) - 1

    def ub(m):
        # greedy maximal packing of disjoint triples -> upper bound on remaining triples
        cnt = 0
        mm = m
        while mm:
            i = (mm & -mm).bit_length() - 1
            bi = B[i] & mm
            if bi == 0:
                break
            j = (bi & -bi).bit_length() - 1
            bj = B[j] & mm
            k = (bj & -bj).bit_length() - 1
            cnt += 1
            mm &= ~((1 << i) | (1 << j) | (1 << k))
        return cnt

    best = [0]

    def rec(m):
        if m == 0:
            if best[0] == 0:
                best[0] = 0
            return
        if _now() > _DEADLINE:
            return
        b = m.bit_count()
        if b <= best[0]:
            return
        if ub(m) == 0:
            if b > best[0]:
                best[0] = b
            return
        # cheap no-triple filter: points that form no triple with any other in m
        safe = 0
        mm = m
        while mm:
            i = (mm & -mm).bit_length() - 1
            bi = B[i] & mm
            if bi == 0:
                safe |= (1 << i)
            mm &= ~(1 << i)
        if safe:
            if b > best[0]:
                best[0] = b
            return
        # branch on a point in a triple, choose the triple with the most "free" points
        mm = m
        best_i = -1
        best_j = -1
        best_k = -1
        best_free = -1
        while mm:
            i = (mm & -mm).bit_length() - 1
            bi = B[i] & mm
            if bi:
                j = (bi & -bi).bit_length() - 1
                bj = B[j] & mm
                k = (bj & -bj).bit_length() - 1
                free = m.bit_count() - 3
                if free > best_free:
                    best_free = free
                    best_i, best_j, best_k = i, j, k
            mm &= ~(1 << i)
        i, j, k = best_i, best_j, best_k
        # exclude i
        rec(m & ~(1 << i))
        # exclude j
        rec(m & ~(1 << j))
        # exclude k
        rec(m & ~(1 << k))

    rec(allmask)
    return best[0]

def _score(pts):
    if _has_four_collinear(pts):
        return None
    B = _build_B(pts)
    return _max_indep(B, 20)

def _seed_points():
    # parabola: no three collinear -> score 20, a valid but high starting point
    return [[x, x * x] for x in range(-10, 10)]

def _candidates():
    c = []
    for x in range(-100, 101):
        for y in range(-100, 101):
            c.append([x, y])
    return c

def _swap(pts, idx, newpt):
    q = [list(p) for p in pts]
    q[idx] = newpt
    return q

def construct():
    global _START, _RNG
    _START = _now()
    _RNG = random.Random(12345)

    pts = _seed_points()
    B = _build_B(pts)
    score = _max_indep(B, 20)

    cand = _candidates()
    # precompute candidate index by (x,y) for fast removal
    cand_set = set((p[0], p[1]) for p in cand)

    # order candidates by how "central" they are (more collinear opportunities)
    # but keep it simple: iterate in a shuffled order
    order = list(range(len(cand)))
    _RNG.shuffle(order)

    improved = True
    while improved and _now() < _DEADLINE:
        improved = False
        # try single swaps
        for idx in range(20):
            if _now() > _DEADLINE:
                break
            cur = tuple(pts[idx])
            for ci in order:
                if _now() > _DEADLINE:
                    break
                np = cand[ci]
                if (np[0], np[1]) == cur:
                    continue
                # quick reject: if new point equals an existing other point
                ok = True
                for j in range(20):
                    if j != idx and pts[j][0] == np[0] and pts[j][1] == np[1]:
                        ok = False
                        break
                if not ok:
                    continue
                q = _swap(pts, idx, np)
                s = _score(q)
                if s is None:
                    continue
                if s < score:
                    pts = q
                    score = s
                    improved = True
                    break
            if improved:
                break
        # if no improvement this pass, do a few random restarts of the local search
        # by perturbing the current best and re-optimizing (limited by time)
        if not improved and _now() < _DEADLINE - 5:
            for _ in range(3):
                if _now() > _DEADLINE - 2:
                    break
                p2 = [list(p) for p in pts]
                for _ in range(3):
                    i = _RNG.randrange(20)
                    ci = _RNG.randrange(len(cand))
                    np = cand[ci]
                    if any(p2[j][0] == np[0] and p2[j][1] == np[1] for j in range(20) if j != i):
                        continue
                    p2[i] = np
                s2 = _score(p2)
                if s2 is not None and s2 < score:
                    pts = p2
                    score = s2
                    improved = True
                    break

    # final safety: ensure validity
    if _has_four_collinear(pts):
        pts = _seed_points()
        score = 20

    return [[int(p[0]), int(p[1])] for p in pts]
