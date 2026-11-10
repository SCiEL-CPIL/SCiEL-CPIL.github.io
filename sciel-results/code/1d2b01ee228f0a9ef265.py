"""Improved search for integral point sets in general position."""
import math
import time
import random
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _gcd3(a, b, c):
    return math.gcd(math.gcd(a, b), c)

def _cross(px, py, qx, qy, rx, ry):
    """Cross product of (q-p) and (r-p)."""
    return (qx - px) * (ry - py) - (qy - py) * (rx - px)

def _no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            for k in range(j + 1, n):
                xk, yk = pts[k]
                if _cross(xi, yi, xj, yj, xk, yk) == 0:
                    return False
    return True

def _no_four_concyclic(pts):
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        a1 = x1*x1 + y1*y1
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            a2 = x2*x2 + y2*y2
            b2 = a2 - a1
            c2 = x2 - x1
            d2 = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = pts[k]
                a3 = x3*x3 + y3*y3
                b3 = a3 - a1
                c3 = x3 - x1
                d3 = y3 - y1
                for l in range(k + 1, n):
                    x4, y4 = pts[l]
                    a4 = x4*x4 + y4*y4
                    b4 = a4 - a1
                    c4 = x4 - x1
                    d4 = y4 - y1
                    det = b2*(c3*d4 - d3*c4) - c2*(b3*d4 - d3*b4) + d2*(b3*c4 - c3*b4)
                    if det == 0:
                        return False
    return True

def _dist_sq_ok(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return _is_square(dx*dx + dy*dy)

def _valid_set(pts):
    n = len(pts)
    if n < 3:
        return False
    if len(set(pts)) != n:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if not _dist_sq_ok(pts[i], pts[j]):
                return False
    if not _no_three_collinear(pts):
        return False
    if not _no_four_concyclic(pts):
        return False
    return True

def _gen_candidates(max_r=300):
    """Generate points at integer distance from origin using Pythagorean triples."""
    cands = set()
    for m in range(2, int(math.sqrt(max_r * max_r)) + 2):
        for n in range(1, m):
            if (m - n) % 2 == 0 or math.gcd(m, n) != 1:
                continue
            a = m*m - n*n
            b = 2*m*n
            c = m*m + n*n
            if c > max_r:
                continue
            for sa in (-1, 1):
                for sb in (-1, 1):
                    cands.add((sa*a, sb*b))
                    cands.add((sa*b, sb*a))
    for r in range(1, max_r + 1):
        cands.add((r, 0))
        cands.add((-r, 0))
        cands.add((0, r))
        cands.add((0, -r))
    return sorted(cands, key=lambda p: p[0]*p[0] + p[1]*p[1])

def construct():
    t0 = time.time()
    deadline = t0 + 140

    # Known 5-point solution
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best = list(base)
    best_n = len(best)

    cands = _gen_candidates(400)
    # Filter: integer distance to all base points
    pre = []
    for c in cands:
        if c in base:
            continue
        ok = True
        for b in base:
            dx = c[0] - b[0]
            dy = c[1] - b[1]
            if not _is_square(dx*dx + dy*dy):
                ok = False
                break
        if ok:
            pre.append(c)

    # Strategy 1: Greedy extension from base
    cur = list(base)
    for c in pre:
        if time.time() > deadline - 5:
            break
        if c in cur:
            continue
        # Quick distance check
        ok = True
        for p in cur:
            dx = c[0] - p[0]
            dy = c[1] - p[1]
            if not _is_square(dx*dx + dy*dy):
                ok = False
                break
        if not ok:
            continue
        new_set = cur + [c]
        if _valid_set(new_set):
            cur = new_set
            if len(cur) > best_n:
                best_n = len(cur)
                best = list(cur)

    # Strategy 2: Backtracking search for 6+ points
    # Start from base, try to add points in all orderings
    def backtrack(pts, remaining, depth, limit):
        if time.time() > deadline - 3:
            return
        if depth > limit:
            return
        for i, c in enumerate(remaining):
            if c in pts:
                continue
            # Check distance to all pts
            ok = True
            for p in pts:
                dx = c[0] - p[0]
                dy = c[1] - p[1]
                if not _is_square(dx*dx + dy*dy):
                    ok = False
                    break
            if not ok:
                continue
            new_pts = pts + [c]
            if _valid_set(new_pts):
                if len(new_pts) > best_n:
                    nonlocal best, best_n
                    best_n = len(new_pts)
                    best = list(new_pts)
                rem = remaining[:i] + remaining[i+1:]
                backtrack(new_pts, rem, depth + 1, limit)
                if time.time() > deadline - 3:
                    return

    # Use a smaller candidate set for backtracking
    bt_cands = pre[:200]
    backtrack(list(base), bt_cands, 0, 3)

    # Strategy 3: Randomized restarts with different seeds
    random.seed(12345)
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        cur = list(base)
        # Shuffle candidates
        shuffled = list(pre)
        random.shuffle(shuffled)
        for c in shuffled:
            if time.time() > deadline - 5:
                break
            if c in cur:
                continue
            ok = True
            for p in cur:
                dx = c[0] - p[0]
                dy = c[1] - p[1]
                if not _is_square(dx*dx + dy*dy):
                    ok = False
                    break
            if not ok:
                continue
            new_set = cur + [c]
            if _valid_set(new_set):
                cur = new_set
                if len(cur) > best_n:
                    best_n = len(cur)
                    best = list(cur)

    # Strategy 4: Local search - swap points
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            improved = False
            cur = list(best)
            for i in range(len(cur)):
                if time.time() > deadline - 10:
                    break
                for c in pre:
                    if c in cur:
                        continue
                    new_set = cur[:i] + cur[i+1:] + [c]
                    if _valid_set(new_set):
                        if len(new_set) > best_n:
                            best_n = len(new_set)
                            best = list(new_set)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    # Final verification
    if _valid_set(best):
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best]}

    return {'D': 1, 'points': [[[0, 1], [0, 1]], [[-56, 1], [33, 1]], [[-16, 1], [63, 1]], [[0, 1], [33, 1]], [[16, 1], [63, 1]]]}
