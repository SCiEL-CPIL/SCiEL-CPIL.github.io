"""Design: pack 20 integer points onto a small set of lines (<=3 per line) so that
every large subset contains a collinear triple, minimizing the largest 3-collinear-free
subset. Construct a base config, validate (distinct, bounds, no-4-collinear), then
local-search by moving single points to candidate spots that preserve validity and
reduce the exact B&B score."""
import math
import time
import random

DEADLINE = time.time() + 135.0   # safety margin under 150s wall-clock
N = 20
BOUND = 100


def _norm(a, b):
    g = math.gcd(abs(a), abs(b))
    if g == 0:
        return (0, 0)
    a //= g
    b //= g
    if a < 0 or (a == 0 and b < 0):
        a, b = -a, -b
    return (a, b)


def _line_key(p1, p2):
    # canonical (normal, c) for the line through p1,p2
    x1, y1 = p1
    x2, y2 = p2
    a = y1 - y2
    b = x2 - x1
    c = -(a * x1 + b * y1)
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    if g == 0:
        g = 1
    a //= g
    b //= g
    c //= g
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)


def _has_four_collinear(pts):
    # True if any line contains >=4 points
    from collections import defaultdict
    d = defaultdict(int)
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            d[_line_key(pts[i], pts[j])] += 1
            if d[_line_key(pts[i], pts[j])] >= 4:
                return True
    return False


def _valid(pts):
    if len(pts) != N:
        return False
    s = set(tuple(p) for p in pts)
    if len(s) != N:
        return False
    for x, y in s:
        if abs(x) > BOUND or abs(y) > BOUND:
            return False
    return not _has_four_collinear(list(s))


def _collinear_triples(pts):
    # list of frozenset of 3 indices that are collinear
    from collections import defaultdict
    d = defaultdict(list)
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(pts[i], pts[j])
            d[k].append((i, j))
    triples = set()
    for k, pairs in d.items():
        # group by line; for each line with m points, all C(m,3) are collinear
        idxs = set()
        for (i, j) in pairs:
            idxs.add(i)
            idxs.add(j)
        idxs = sorted(idxs)
        m = len(idxs)
        if m >= 3:
            for a in range(m):
                for b in range(a + 1, m):
                    for c in range(b + 1, m):
                        triples.add(frozenset((idxs[a], idxs[b], idxs[c])))
    return list(triples)


def _max_independent(pts):
    # largest subset of points with no 3 collinear (exact branch and bound)
    n = len(pts)
    triples = _collinear_triples(pts)
    # for each point, the set of other points it forms a triple with (partner pairs)
    # We need: a subset S is valid iff no triple is fully contained in S.
    # Represent each triple as a frozenset of 3 indices.
    # Branch and bound over points, tracking which triples are "blocked" (have >=2 in S).
    # A triple becomes invalid (forces exclusion) once it has 3 in S.
    # Prune: if a triple has 2 in S and we are about to add the 3rd -> forbidden.
    # Also upper bound: current size + remaining points.

    triples_by_point = [[] for _ in range(n)]
    for t in triples:
        for p in t:
            triples_by_point[p].append(t)

    best = [0]
    order = list(range(n))

    # state: selected list, blocked set (triples with >=2 selected), and index
    # We track for each triple its count in selected.
    # To be fast, use counts array.
    import sys
    sys.setrecursionlimit(10000)

    counts = [0] * len(triples)
    t_index = {}
    for i, t in enumerate(triples):
        t_index[t] = i

    def rec(pos, size):
        if size + (n - pos) <= best[0]:
            return
        if pos == n:
            if size > best[0]:
                best[0] = size
            return
        p = order[pos]
        # Option 1: skip p
        rec(pos + 1, size)
        # Option 2: take p if it doesn't complete any triple
        ok = True
        changed = []
        for t in triples_by_point[p]:
            ti = t_index[t]
            if counts[ti] == 2:
                ok = False
                break
        if ok:
            for t in triples_by_point[p]:
                ti = t_index[t]
                counts[ti] += 1
                changed.append(ti)
            rec(pos + 1, size + 1)
            for ti in changed:
                counts[ti] -= 1

    rec(0, 0)
    return best[0]


def _base_config():
    # Build 20 points on lines of size <=3, spread in the box.
    # Use a set of lines; place points as integer intersections.
    # Simple constructive approach: take a 5x4 grid subset but perturb to avoid 4-collinear,
    # then rely on local search. Instead, start from a known dense collinear arrangement.
    # Lines (each with 3 points):
    lines = [
        [(0, 0), (1, 0), (2, 0)],
        [(0, 1), (1, 1), (2, 1)],
        [(0, 2), (1, 2), (2, 2)],
        [(0, 3), (1, 3), (2, 3)],
        [(0, 4), (1, 4), (2, 4)],
    ]
    pts = []
    for L in lines:
        pts.extend(L)
    # 15 points; need 5 more. Add points on new lines (size 1 or 2) that also create triples
    # with existing points to increase density.
    extra = [
        (3, 0),   # collinear with (0,0),(1,0)? line y=0 -> (0,0),(1,0),(2,0),(3,0) = 4! bad
    ]
    # Avoid 4 on y=0. Instead add points forming new triples.
    extra = [
        (3, 1),   # line y=1 has (0,1),(1,1),(2,1),(3,1)=4 bad
    ]
    # The 5x3 grid rows each have 3; adding a 4th in a row makes 4. So keep rows at 3.
    # Add points in new rows/columns that create diagonal triples.
    extra = [
        (3, 2),   # check line with (1,1),(2,? ) ... (3,2),(2,1),(1,0): slope 1 -> (1,0),(2,1),(3,2) triple! and (3,2) row y=2 has (0,2),(1,2),(2,2),(3,2)=4 bad
    ]
    # This is getting fiddly. Use a cleaner construction: 20 points as a 4x5 grid is 4 per row -> invalid.
    # Use points on the parabola then local search to introduce collinearities? Parabola has none.
    # Better: start from a random valid config and local search.
    # Construct a valid random config: sample points, reject if 4-collinear or duplicate.
    rng = random.Random(12345)
    cand = set()
    allpts = set()
    while len(cand) < N:
        x = rng.randint(-BOUND, BOUND)
        y = rng.randint(-BOUND, BOUND)
        p = (x, y)
        if p in cand:
            continue
        trial = cand | {p}
        # quick 4-collinear check on the new point with pairs of existing
        ok = True
        from collections import defaultdict
        d = defaultdict(int)
        lst = list(trial)
        # only check lines through new point
        for q in lst:
            if q == p:
                continue
            d[_line_key(p, q)] += 1
            if d[_line_key(p, q)] >= 4:
                ok = False
                break
        if ok:
            cand.add(p)
    return list(cand)


def _candidate_moves(pts):
    # candidate positions to move a point to: all grid points in a small window around each point
    # plus a global random sample.
    cands = set()
    for (x, y) in pts:
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nx, ny = x + dx, y + dy
                if -BOUND <= nx <= BOUND and -BOUND <= ny <= BOUND:
                    cands.add((nx, ny))
    return list(cands)


def construct():
    start = time.time()
    rng = random.Random(7)
    pts = _base_config()
    if not _valid(pts):
        # fallback: parabola (valid, score 20)
        return [[x, x * x] for x in range(-10, 10)]
    best_score = _max_independent(pts)
    best_pts = [list(p) for p in pts]
    cands = _candidate_moves(pts)

    while time.time() < DEADLINE:
        improved = False
        for i in range(N):
            if time.time() > DEADLINE:
                break
            old = pts[i]
            for np_ in cands:
                if np_ == old:
                    continue
                if np_ in pts and np_ != old:
                    # would duplicate; skip unless it's the same point
                    continue
                trial = list(pts)
                trial[i] = np_
                if _valid(trial):
                    sc = _max_independent(trial)
                    if sc < best_score:
                        best_score = sc
                        best_pts = [list(p) for p in trial]
                        pts = trial
                        improved = True
                        cands = _candidate_moves(pts)
                        break
        if not improved:
            # perturb: random restart with a different seed to escape local minima
            newpts = _base_config()
            if _valid(newpts):
                sc = _max_independent(newpts)
                if sc < best_score:
                    best_score = sc
                    best_pts = [list(p) for p in newpts]
                    pts = newpts
                    cands = _candidate_moves(pts)
                # else keep current
                # try a few random restarts
                for _ in range(3):
                    if time.time() > DEADLINE:
                        break
                    rp = _base_config()
                    if _valid(rp):
                        sc = _max_independent(rp)
                        if sc < best_score:
                            best_score = sc
                            best_pts = [list(p) for p in rp]
                            pts = rp
                            cands = _candidate_moves(pts)

    return best_pts


if __name__ == "__main__":
    res = construct()
    assert _valid(res), "invalid configuration"
    print("score:", _max_independent(res))
