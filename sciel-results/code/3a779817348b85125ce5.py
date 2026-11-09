"""Contact numbers of 32 unit pennies on the triangular lattice (Harborth optimum = 76)."""
import time

# Hard wall-clock guard (well under the 150 s kill, with a safety margin).
_START = time.time()
_DEADLINE = 130.0


def _sq(p, q):
    a, b = p
    c, d = q
    return a * a + b * b - a * c - a * d - b * c + b * d + c * c + d * d


def _count_unit(pts):
    n = len(pts)
    c = 0
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            if _sq(pi, pts[j]) == 1:
                c += 1
    return c


def _valid(pts):
    if len(pts) != 32:
        return False
    if len(set(tuple(p) for p in pts)) != 32:
        return False
    for p in pts:
        if abs(p[0]) > 64 or abs(p[1]) > 64:
            return False
    return True


def _hexagon(radius):
    pts = []
    for q in range(-radius, radius + 1):
        r_min = max(-radius, -q - radius)
        r_max = min(radius, -q + radius)
        for r in range(r_min, r_max + 1):
            pts.append((q, -q - r))
    return pts


def _ring(radius):
    pts = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if abs(q + r) <= radius and max(abs(q), abs(r), abs(q + r)) == radius:
                pts.append((q, -q - r))
    return pts


def _build_base():
    pts = _hexagon(2)                      # 19 cells (radius-2 hexagon)
    if len(pts) >= 32:
        return pts[:32]
    ring3 = _ring(3)                       # 18 cells
    S = set(pts)
    for c in ring3:
        if len(pts) >= 32:
            break
        # number of already-present neighbors (more = more interior = better)
        nb = sum(1 for d in ((1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1))
                 if (c[0] + d[0], c[1] + d[1]) in S)
        pts.append(c)
        S.add(c)
    return pts[:32]


def _local_search(pts, budget=110.0):
    pts = [list(p) for p in pts]
    n = len(pts)
    best = _count_unit(pts)

    def time_ok():
        return time.time() < _DEADLINE and (time.time() - _START) < budget

    improved = True
    while improved and time_ok():
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if not time_ok():
                    return pts, best
                # candidate swap: put cell j into slot i's position and vice versa
                a, b = pts[i]
                c, d = pts[j]
                # new i = (c,d), new j = (a,b); check distinctness & bounds
                # (positions are reused, so bounds are always fine; distinctness
                #  is preserved because we only swap two existing cells)
                pts[i] = [c, d]
                pts[j] = [a, b]
                cnt = _count_unit(pts)
                if cnt > best:
                    best = cnt
                    improved = True
                else:
                    pts[i] = [a, b]
                    pts[j] = [c, d]
    return pts, best


def construct():
    pts = _build_base()
    if not _valid(pts):
        # Fallback: a guaranteed-valid compact cluster.
        pts = _hexagon(2) + _ring(3)[:13]
    score = _count_unit(pts)
    if score < 76 and time.time() < _DEADLINE:
        pts, score = _local_search(pts)
    if not _valid(pts):
        pts = _build_base()
    # Return as a list of two-element lists, exactly 32 distinct pairs.
    return [[int(a), int(b)] for (a, b) in pts[:32]]
