import time
import sys
from math import gcd
from itertools import combinations

def _count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points."""
    n = len(points)
    lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            g = gcd(abs(a), gcd(abs(b), abs(c)))
            if g > 0:
                a //= g
                b //= g
                c //= g
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a, b, c = -a, -b, -c
            key = (a, b, c)
            if key not in lines:
                lines[key] = 0
            lines[key] += 1
    return sum(1 for v in lines.values() if v == 1)

def _is_valid(points):
    if len(points) != 12:
        return False
    if len(set(tuple(p) for p in points)) != 12:
        return False
    # Not all collinear
    x0, y0 = points[0]
    x1, y1 = points[1]
    for k in range(2, 12):
        xk, yk = points[k]
        if (y1 - y0) * (xk - x0) != (yk - y0) * (x1 - x0):
            return True
    return False

def construct():
    deadline = time.time() + 120.0

    # Try several known configurations
    candidates = []

    # 11 on x-axis, 1 off: 11 ordinary lines
    candidates.append([[i, 0] for i in range(11)] + [[0, 1]])

    # 3x4 grid
    candidates.append([[x, y] for x in range(4) for y in range(3)])

    # 4x3 grid
    candidates.append([[x, y] for x in range(3) for y in range(4)])

    # 2x6 grid
    candidates.append([[x, y] for x in range(6) for y in range(2)])

    # Two parallel lines with 6 each
    candidates.append([[x, 0] for x in range(6)] + [[x, 1] for x in range(6)])

    # Triangle with 4 on each edge (sharing vertices)
    # (0,0),(1,0),(2,0),(3,0), (0,1),(0,2),(0,3), (1,2),(2,1)
    # That's 9, need 3 more. Add (1,1),(2,2),(3,3)? No, let me think.
    # Actually: line1: y=0: (0,0),(1,0),(2,0),(3,0)
    # line2: x=0: (0,0),(0,1),(0,2),(0,3)
    # line3: x+y=3: (0,3),(1,2),(2,1),(3,0)
    # Unique: (0,0),(1,0),(2,0),(3,0),(0,1),(0,2),(0,3),(1,2),(2,1) = 9 points
    # Add 3 more: (1,1),(2,2),(1,3) or similar
    candidates.append([[0,0],[1,0],[2,0],[3,0],[0,1],[0,2],[0,3],[1,2],[2,1],[1,1],[2,2],[1,3]])

    # Hexagonal arrangement
    candidates.append([[3,0],[0,3],[-3,0],[0,-3],[-3,-3],[3,-3],[1,2],[-1,2],[2,-1],[-2,-1],[1,-2],[-1,-2]])

    # 4 points on y=0, 4 on y=1, 4 on y=2, staggered
    candidates.append([[0,0],[1,0],[2,0],[3,0],[1,1],[2,1],[3,1],[4,1],[2,2],[3,2],[4,2],[5,2]])

    # Evaluate candidates
    best_pts = None
    best_count = float('inf')
    for pts in candidates:
        if not _is_valid(pts):
            continue
        cnt = _count_ordinary_lines(pts)
        if cnt < best_count:
            best_count = cnt
            best_pts = pts[:]

    if best_pts is None:
        return candidates[0]

    # Local search: move individual points
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(12):
            if time.time() > deadline - 10.0:
                break
            x_orig, y_orig = best_pts[i]
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = x_orig + dx
                    new_y = y_orig + dy
                    if abs(new_x) > 10**9 or abs(new_y) > 10**9:
                        continue
                    new_pts = best_pts[:]
                    new_pts[i] = [new_x, new_y]
                    if not _is_valid(new_pts):
                        continue
                    cnt = _count_ordinary_lines(new_pts)
                    if cnt < best_count:
                        best_count = cnt
                        best_pts = new_pts
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return best_pts
