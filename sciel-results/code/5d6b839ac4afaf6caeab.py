"""Rewritten program: maximize largest multiplicity of a positive doubled area among 30 points."""
import time
import random
from collections import Counter

def _doubled_area(p, q, r):
    ax, ay = q[0] - p[0], q[1] - p[1]
    bx, by = r[0] - p[0], r[1] - p[1]
    return abs(ax * by - ay * bx)

def _score(points):
    areas = Counter()
    n = len(points)
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                a = _doubled_area(pi, pj, pk)
                if a > 0:
                    areas[a] += 1
    if not areas:
        return 0
    return max(areas.values())

def _valid(points):
    if len(points) != 30:
        return False
    s = set(points)
    if len(s) != 30:
        return False
    for x, y in points:
        if not (-30 <= x <= 30 and -30 <= y <= 30):
            return False
    return True

def _grid_search(deadline):
    best_score = 0
    best_pts = None
    for mx in range(1, 31):
        for my in range(1, 31):
            if mx * my != 30:
                continue
            for dx in range(-30, 31):
                for dy in range(-30, 31):
                    if dx == 0 and dy == 0:
                        continue
                    x_min_off = min(0, (mx - 1) * dx)
                    x_max_off = max(0, (mx - 1) * dx)
                    y_min_off = min(0, (my - 1) * dy)
                    y_max_off = max(0, (my - 1) * dy)
                    base_x_min = -30 - x_min_off
                    base_x_max = 30 - x_max_off
                    base_y_min = -30 - y_min_off
                    base_y_max = 30 - y_max_off
                    if base_x_min > base_x_max or base_y_min > base_y_max:
                        continue
                    bx0 = (base_x_min + base_x_max) // 2
                    by0 = (base_y_min + base_y_max) // 2
                    pts = []
                    for i in range(mx):
                        for j in range(my):
                            pts.append((bx0 + i * dx, by0 + j * dy))
                    if not _valid(pts):
                        continue
                    sc = _score(pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = pts[:]
                    if time.time() > deadline:
                        return best_score, best_pts
    return best_score, best_pts

def _try_hexagonal(deadline):
    best_score = 0
    best_pts = None
    for offset in range(0, 2):
        for base_x in range(-30, 31):
            for base_y in range(-30, 31):
                pts = []
                for j in range(10):
                    row_x = base_x + (j % 2) * offset
                    for i in range(3):
                        x = row_x + i * 2
                        y = base_y + j * 1
                        if -30 <= x <= 30 and -30 <= y <= 30:
                            pts.append((x, y))
                if len(pts) < 30:
                    continue
                pts = pts[:30]
                if not _valid(pts):
                    continue
                sc = _score(pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = pts[:]
                if time.time() > deadline:
                    return best_score, best_pts
    return best_score, best_pts

def _local_improve(points, deadline):
    best_pts = points[:]
    best_score = _score(best_pts)
    all_points = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    current_set = set(best_pts)
    random.shuffle(all_points)
    for idx in range(len(best_pts)):
        for candidate in all_points:
            if candidate in current_set:
                continue
            new_pts = best_pts[:]
            new_pts[idx] = candidate
            sc = _score(new_pts)
            if sc > best_score:
                best_score = sc
                best_pts = new_pts
                current_set = set(best_pts)
            if time.time() > deadline:
                return best_score, best_pts
    return best_score, best_pts

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_score = 0
    best_pts = None
    
    score, pts = _grid_search(deadline)
    if pts:
        best_score = score
        best_pts = pts
        if time.time() > deadline:
            return best_pts
    
    score, pts = _try_hexagonal(deadline)
    if pts and score > best_score:
        best_score = score
        best_pts = pts
        if time.time() > deadline:
            return best_pts
    
    if best_pts:
        score, pts = _local_improve(best_pts, deadline)
        if pts and score > best_score:
            best_score = score
            best_pts = pts
    
    if best_pts is None:
        return [(x, y) for x in range(6) for y in range(5)]
    
    return best_pts
# EVOLVE-BLOCK-END
