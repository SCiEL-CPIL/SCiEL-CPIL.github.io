"""SCiEL-evolved program: line-packing heuristic + exact MIS branch-and-bound scorer."""
import time
import itertools
import random
from fractions import Fraction

_START = time.time()
_DEADLINE = 135.0  # safety margin before the 150s kill

def _norm(v):
    x, y = v
    g = abs(x)
    if g == 0:
        g = abs(y)
    if g == 0:
        return (0, 0)
    if x < 0 or (x == 0 and y < 0):
        g = -g
    return (x // g, y // g)

def _line_key(p, q):
    # canonical key for the line through p and q
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    n = _norm((dy, -dx))
    # c = n.x * x + n.y * y ; normalize sign of c
    c = n[0] * p[0] + n[1] * p[1]
    if c < 0:
        c = -c
    return (n[0], n[1], c)

def _triple_lines(points):
    # return dict line_key -> set of point indices on that line (only lines with >=3)
    lines = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            lines.setdefault(k, set()).add(i)
            lines[k].add(j)
    return {k: v for k, v in lines.items() if len(v) >= 3}

def _triple_sets(points):
    # return list of frozensets (each size>=3) of point indices that are collinear
    lines = _triple_lines(points)
    return [frozenset(v) for v in lines.values()]

def _is_valid(points):
    # no four collinear
    for v in _triple_lines(points).values():
        if len(v) >= 4:
            return False
    return True

def _score(points):
    # exact largest subset with no three collinear = max independent set of the
    # "conflict" graph where an edge joins two points if they lie on a shared line
    # with a third point (i.e. their removal is forced together). We model it as:
    # a subset S is valid (no three collinear) iff no collinear triple is fully
    # contained in S. We find the max S via branch and bound on the hypergraph.
    n = len(points)
    triples = _triple_sets(points)
    # Build conflict: for each point, the set of points it shares a line-with-3 with
    # We'll do a simple exact branch and bound over inclusion/exclusion.
    # Represent each triple; a solution is a set of indices with no triple subset.
    # Use B&B: choose a triple, branch on excluding each of its members.
    # Upper bound = number of remaining candidate points.

    best = [0]
    cand = list(range(n))

    # Precompute for speed: for each index, list of triples containing it
    triples_by_pt = [[] for _ in range(n)]
    for t in triples:
        for i in t:
            triples_by_pt[i].append(t)

    def rec(remaining):
        if not remaining:
            if len(remaining) > best[0]:
                best[0] = len(remaining)
            return
        # upper bound
        if len(remaining) <= best[0]:
            return
        # find a triple fully inside remaining
        remset = set(remaining)
        triple = None
        for i in remaining:
            for t in triples_by_pt[i]:
                if t <= remset:
                    triple = t
                    break
            if triple is not None:
                break
        if triple is None:
            # no three collinear in remaining -> all can be taken
            val = len(remaining)
            if val > best[0]:
                best[0] = val
            return
        # branch: exclude each member of the triple
        for m in triple:
            new_rem = [x for x in remaining if x != m]
            rec(new_rem)

    rec(cand)
    return best[0]

def _random_config(seed):
    rng = random.Random(seed)
    pts = []
    seen = set()
    # pick points from a grid, biased to create collinear triples but avoid 4-collinear
    while len(pts) < 20:
        x = rng.randint(-50, 50)
        y = rng.randint(-50, 50)
        if (x, y) in seen:
            continue
        seen.add((x, y))
        # check no four collinear with existing
        ok = True
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                k = _line_key(pts[i], pts[j])
                # count how many existing points are on this line
                cnt = 0
                for p in pts:
                    if _line_key(pts[i], p) == k or _line_key(pts[j], p) == k:
                        cnt += 1
                # simpler: check new point against all pairs
                pass
        # robust check: ensure no line has >=4 including new point
        pts.append((x, y))
        lines = _triple_lines(pts)
        if any(len(v) >= 4 for v in lines.values()):
            pts.pop()
            continue
    return [[p[0], p[1]] for p in pts]

def _line_packing_config(seed):
    # constructive: pick several lines, place 3 points each, ensure no line gets 4
    rng = random.Random(seed)
    pts = []
    seen = set()
    used_lines = {}
    attempts = 0
    while len(pts) < 20 and attempts < 5000:
        attempts += 1
        # choose a random line
        x0 = rng.randint(-40, 40)
        y0 = rng.randint(-40, 40)
        dx = rng.randint(-5, 5)
        dy = rng.randint(-5, 5)
        if dx == 0 and dy == 0:
            continue
        key = _norm((dx, dy))
        # sample 3 points on this line
        new_on_line = []
        for t in range(rng.randint(0, 5), rng.randint(6, 15)):
            x = x0 + dx * t
            y = y0 + dy * t
            if abs(x) > 100 or abs(y) > 100:
                continue
            if (x, y) in seen:
                continue
            new_on_line.append((x, y))
            if len(new_on_line) >= 3:
                break
        if len(new_on_line) < 3:
            continue
        # check adding these doesn't create a 4-collinear anywhere
        trial = pts + new_on_line
        lines = _triple_lines(trial)
        if any(len(v) >= 4 for v in lines.values()):
            continue
        pts = trial
        for p in new_on_line:
            seen.add(p)
    if len(pts) < 20:
        return None
    return [[p[0], p[1]] for p in pts[:20]]

def construct():
    best_pts = None
    best_score = 10**9
    start = time.time()
    # deterministic baseline first
    base = _line_packing_config(12345)
    if base is None:
        base = _random_config(12345)
    if base is not None and _is_valid(base):
        s = _score(base)
        best_pts = base
        best_score = s
    # search
    seed = 0
    while time.time() - start < _DEADLINE - 5:
        seed += 1
        cfg = _line_packing_config(seed)
        if cfg is None:
            cfg = _random_config(seed)
        if cfg is None or not _is_valid(cfg):
            continue
        s = _score(cfg)
        if s < best_score:
            best_score = s
            best_pts = cfg
            if best_score <= 10:
                break
    if best_pts is None:
        # fallback parabola (valid, score 20)
        best_pts = [[x, x * x] for x in range(-10, 10)]
    return best_pts
