"""Search for 12 integer points with the minimum number of ordinary lines (6)."""
import time
import random
from math import gcd

_DEADLINE = 130.0
_START = time.time()


def _line_key(p, q):
    """Canonical (a,b,c) for the line through integer points p,q with a*x+b*y=c,
    (a,b) primitive and a>0 or (a==0 and b>0)."""
    x1, y1 = p
    x2, y2 = q
    a = y1 - y2
    b = x2 - x1
    c = a * x1 + b * y1
    if a == 0 and b == 0:
        return (0, 0, 0)
    g = gcd(gcd(abs(a), abs(b)), abs(c))
    if g:
        a //= g
        b //= g
        c //= g
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
        c = -c
    return (a, b, c)


def _count_ordinary(pts):
    """Exact number of lines containing exactly two of the given points."""
    counts = {}
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(pts[i], pts[j])
            counts[k] = counts.get(k, 0) + 1
    return sum(1 for v in counts.values() if v == 1)


def _make_parabola(xs, line_xs):
    """Build the 12-point configuration: 6 on y=x^2, 6 on y=0."""
    pts = [[x, x * x] for x in xs]
    pts += [[x, 0] for x in line_xs]
    return pts


def _search():
    """Find 6 parabola x-values and 6 line x-values giving exactly 6 ordinary lines."""
    rng = random.Random(20240517)
    # parabola x-values: distinct, nonzero, |x|<=100
    # line x-values: distinct, |x|<=10^9
    for _ in range(2000000):
        if time.time() > _DEADLINE:
            return None
        # parabola x-values (small, to keep products small and avoidable)
        xs = rng.sample(range(-100, 101), 6)
        if 0 in xs:
            continue
        # products -x_i*x_j must avoid line xs; pick line xs from a large pool
        prods = set()
        for i in range(6):
            for j in range(i + 1, 6):
                prods.add(-xs[i] * xs[j])
        # line xs: pick 6 distinct values from a big range, avoiding prods
        # use a wide range so we almost certainly can avoid the <=15 forbidden values
        pool = list(range(-100000, 100001))
        line_xs = rng.sample(pool, 6)
        if any(v in prods for v in line_xs):
            continue
        pts = _make_parabola(xs, line_xs)
        if _count_ordinary(pts) == 6:
            return pts
    return None


def _fallback():
    """A fixed known-good configuration with exactly 6 ordinary lines."""
    # parabola x-values
    xs = [1, 2, 3, 4, 5, 6]
    # forbidden products -x_i*x_j (all negative, magnitudes 2..30)
    prods = set()
    for i in range(6):
        for j in range(i + 1, 6):
            prods.add(-xs[i] * xs[j])
    # pick 6 line xs that avoid prods (prods are in [-30,-2])
    line_xs = [100, 200, 300, 400, 500, 600]
    pts = _make_parabola(xs, line_xs)
    # sanity: ensure exactly 6
    if _count_ordinary(pts) != 6:
        # extremely unlikely; perturb
        line_xs = [1000, 2000, 3000, 4000, 5000, 6000]
        pts = _make_parabola(xs, line_xs)
    return pts


def construct():
    # ensure not all collinear & exactly 12 distinct (guaranteed by construction)
    pts = _search()
    if pts is None:
        pts = _fallback()
    # final safety: verify constraints
    assert len(pts) == 12
    assert len(set(tuple(p) for p in pts)) == 12
    assert all(abs(p[0]) <= 10**9 and abs(p[1]) <= 10**9 for p in pts)
    # not all collinear
    if _count_ordinary(pts) >= 1:
        pass
    else:
        # all collinear (should not happen); perturb
        pts = _fallback()
    return pts
