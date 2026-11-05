"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

_start_time = time.time()
_DEADLINE = 130  # safety margin below the 150s wall-clock kill

N = 16
BOUND = 1000

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a if a else 1

def _line_key(p, q):
    """Canonical (slope, intercept) key for the line through p and q."""
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    g = _gcd(abs(dx), abs(dy))
    if g > 1:
        dx //= g
        dy //= g
    # intercept b = p[1]*dx - p[0]*dy, normalized sign
    b = p[1] * dx - p[0] * dy
    if b < 0:
        b = -b
    return (dx, dy, b)

def _build(pts):
    """Build per-point line structure: lines[i] = {key: frozenset(indices)}."""
    n = len(pts)
    lines = [dict() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(pts[i], pts[j])
            lines[i].setdefault(k, set()).add(i)
            lines[i][k].add(j)
            lines[j].setdefault(k, set()).add(i)
            lines[j][k].add(j)
    # freeze to frozenset for hashing / set membership
    for i in range(n):
        for k in lines[i]:
            lines[i][k] = frozenset(lines[i][k])
    return lines

def _score(lines):
    """Count lines containing exactly 4 points (unique, via index set)."""
    seen = set()
    for i in range(N):
        for k, s in lines[i].items():
            if len(s) == 4:
                seen.add(s)
    return len(seen)

def _valid(pts):
    if len(pts) != N:
        return False
    for x, y in pts:
        if abs(x) > BOUND or abs(y) > BOUND:
            return False
    if len(set(tuple(p) for p in pts)) != N:
        return False
    lines = _build(pts)
    for i in range(N):
        for s in lines[i].values():
            if len(s) >= 5:
                return False
    return True

def _move_point(pts, lines, idx, dx, dy):
    """Move point idx by (dx,dy). Return (new_pts, new_lines) or None if invalid.

    Incremental: only the 15 lines through idx change. All other lines keep
    their exact membership, so validity is preserved incrementally."""
    x, y = pts[idx][0] + dx, pts[idx][1] + dy
    if abs(x) > BOUND or abs(y) > BOUND:
        return None
    new_pts = [list(p) for p in pts]
    for j in range(N):
        if j != idx and new_pts[j] == [x, y]:
            return None
    new_pts[idx] = [x, y]

    # Old lines through idx (as dict key->frozenset)
    old = lines[idx]
    # New lines through idx
    new = dict()
    for j in range(N):
        if j == idx:
            continue
        k = _line_key(new_pts[idx], new_pts[j])
        new.setdefault(k, set()).add(idx)
        new[k].add(j)
    # Any new line with 5+ points -> invalid
    for k, s in new.items():
        if len(s) >= 5:
            return None

    new_lines = [None] * N
    for i in range(N):
        if i == idx:
            new_lines[i] = {k: frozenset(s) for k, s in new.items()}
        else:
            d = dict(lines[i])
            for k in list(d.keys()):
                if idx in d[k]:
                    del d[k]
            if idx in new:
                d[_line_key(new_pts[idx], new_pts[i])] = frozenset(new[_line_key(new_pts[idx], new_pts[i])])
            new_lines[i] = d
    return new_pts, new_lines

def _anneal(pts, lines, score, deadline, T0=8.0, T_min=0.02, cool=0.9995):
    """Simulated annealing with periodic escape jumps. Returns best found."""
    cur_pts, cur_lines, cur_score = pts, lines, score
    best_pts, best_lines, best_score = [list(p) for p in pts], lines, score
    T = T0
    it = 0
    while time.time() < deadline and T > T_min:
        it += 1
        idx = random.randint(0, N - 1)
        if it % 12 == 0:
            # plateau / escape move: larger jump to leave the local basin
            dx = random.randint(-40, 40)
            dy = random.randint(-40, 40)
        else:
            dx = random.randint(-8, 8)
            dy = random.randint(-8, 8)
        if dx == 0 and dy == 0:
            continue
        r = _move_point(cur_pts, cur_lines, idx, dx, dy)
        if r is None:
            T *= cool
            continue
        npts, nlines = r
        nscore = _score(nlines)
        delta = nscore - cur_score
        if delta > 0 or random.random() < math.exp(delta / T):
            cur_pts, cur_lines, cur_score = npts, nlines, nscore
            if cur_score > best_score:
                best_pts, best_lines, best_score = [list(p) for p in cur_pts], cur_lines, cur_score
        T *= cool
    return best_pts, best_lines, best_score

def _local_polish(pts, lines, score, deadline):
    """Exhaustive single-point polish: try every (dx,dy) for every point."""
    cur_pts, cur_lines, cur_score = pts, lines, score
    best_pts, best_score = [list(p) for p in pts], score
    for idx in range(N):
        if time.time() > deadline:
            break
        cx, cy = cur_pts[idx][0], cur_pts[idx][1]
        for dx in range(-20, 21):
            if time.time() > deadline:
                break
            for dy in range(-20, 21):
                if dx == 0 and dy == 0:
                    continue
                if abs(cx + dx) > BOUND or abs(cy + dy) > BOUND:
                    continue
                r = _move_point(cur_pts, cur_lines, idx, dx, dy)
                if r is None:
                    continue
                npts, nlines = r
                nscore = _score(nlines)
                if nscore > cur_score:
                    cur_pts, cur_lines, cur_score = npts, nlines, nscore
                    if cur_score > best_score:
                        best_pts, best_score = [list(p) for p in cur_pts], cur_score
    return best_pts, best_score

def construct():
    global _start_time
    random.seed(42)
    deadline = _start_time + _DEADLINE

    # Baseline: 4x4 grid = affine plane AG(2,4), 24 four-point lines.
    base = [[x, y] for x in range(4) for y in range(4)]
    base_lines = _build(base)
    base_score = _score(base_lines)

    best_pts, best_lines, best_score = base, base_lines, base_score

    # Phase 1: anneal from the grid
    if time.time() < deadline - 5:
        p, l, s = _anneal(base, base_lines, base_score, min(deadline, _start_time + 45))
        if s > best_score:
            best_pts, best_lines, best_score = p, l, s

    # Phase 2: anneal from perturbed grids (start outside the grid basin)
    for trial in range(40):
        if time.time() > deadline - 20:
            break
        pts = [list(p) for p in base]
        for _ in range(random.randint(1, 4)):
            idx = random.randint(0, N - 1)
            pts[idx] = [pts[idx][0] + random.randint(-4, 4),
                        pts[idx][1] + random.randint(-4, 4)]
        if not _valid(pts):
            continue
        lines = _build(pts)
        sc = _score(lines)
        # short anneal per restart
        p, l, s = _anneal(pts, lines, sc, min(deadline, _start_time + 45 + 1.5 * trial),
                          T0=4.0, T_min=0.02, cool=0.999)
        if s > best_score:
            best_pts, best_lines, best_score = p, l, s

    # Phase 3: exhaustive polish on the best configuration
    if time.time() < deadline - 3:
        p, s = _local_polish(best_pts, best_lines, best_score, deadline)
        if s > best_score:
            best_pts, best_score = p, s

    return [[int(x), int(y)] for x, y in best_pts]

# EVOLVE-BLOCK-END
