import numpy as np
import time
from math import gcd

def _normalize_line(A, B, C):
    if A == 0 and B == 0 and C == 0:
        return (0, 0, 0)
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _build_grid(cols, rows):
    xs = np.arange(cols)
    ys = np.arange(rows)
    PX, PY = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([PX.ravel(), PY.ravel()], axis=1).astype(np.int64)
    line_set = set()
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            norm = _normalize_line(A, B, C)
            line_set.add(norm)
    lines_list = list(line_set)
    counts = np.array([int(np.count_nonzero(A * points[:, 0] + B * points[:, 1] + C == 0)) for A, B, C in lines_list])
    top = np.argsort(-counts)[:27]
    lines = np.array([lines_list[i] for i in top], dtype=np.int64)
    return points, lines

def _anneal(points, lines, deadline, seed=0):
    points = points.copy()
    lines = lines.copy()
    A = lines[:, 0:1]
    B = lines[:, 1:2]
    C = lines[:, 2:3]
    x = points[:, 0:1].T
    y = points[:, 1:2].T
    M = (A * x + B * y + C == 0)
    score = int(np.sum(M))
    best_score = score
    best_points = points.copy()
    best_lines = lines.copy()
    line_norms = [_normalize_line(int(lines[i][0]), int(lines[i][1]), int(lines[i][2])) for i in range(27)]
    rng = np.random.default_rng(seed)
    T = 3.0
    cool = 0.99997
    point_set = set(map(tuple, points.tolist()))
    
    while time.time() < deadline:
        if rng.random() < 0.65:
            pi = int(rng.integers(0, 54))
            ox, oy = points[pi]
            for _ in range(8):
                dx = int(rng.integers(-5, 6))
                dy = int(rng.integers(-5, 6))
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                if (nx, ny) in point_set:
                    continue
                new_col_arr = (A * nx + B * ny + C).ravel()
                new_col = int(np.count_nonzero(new_col_arr == 0))
                old_col = int(np.count_nonzero(M[:, pi]))
                delta = new_col - old_col
                if delta >= 0 or rng.random() < np.exp(delta / max(T, 0.01)):
                    point_set.discard((ox, oy))
                    point_set.add((nx, ny))
                    points[pi] = [nx, ny]
                    M[:, pi] = (new_col_arr == 0).astype(bool)
                    score += delta
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                        best_lines = lines.copy()
                    break
        else:
            li = int(rng.integers(0, 27))
            p1 = int(rng.integers(0, 54))
            p2 = int(rng.integers(0, 54))
            if p1 == p2:
                continue
            x1, y1 = points[p1]
            x2, y2 = points[p2]
            A2 = y2 - y1
            B2 = x1 - x2
            C2 = x2 * y1 - x1 * y2
            if A2 == 0 and B2 == 0:
                continue
            norm = _normalize_line(A2, B2, C2)
            dup = False
            for i in range(27):
                if i != li and line_norms[i] == norm:
                    dup = True
                    break
            if dup:
                continue
            new_row_arr = (A2 * points[:, 0] + B2 * points[:, 1] + C2)
            new_row = int(np.count_nonzero(new_row_arr == 0))
            old_row = int(np.count_nonzero(M[li, :]))
            delta = new_row - old_row
            if delta >= 0 or rng.random() < np.exp(delta / max(T, 0.01)):
                lines[li] = norm
                line_norms[li] = norm
                M[li, :] = (new_row_arr == 0).astype(bool)
                score += delta
                if score > best_score:
                    best_score = score
                    best_points = points.copy()
                    best_lines = lines.copy()
        T *= cool
    return best_points, best_lines, best_score

def _is_valid(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    if len(set(map(tuple, points.tolist()))) != 54:
        return False
    ls = set()
    for i in range(27):
        A, B, C = int(lines[i][0]), int(lines[i][1]), int(lines[i][2])
        if A == 0 and B == 0:
            return False
        n = _normalize_line(A, B, C)
        if n in ls:
            return False
        ls.add(n)
    return True

def szemeredi_trotter_incidences():
    start = time.time()
    deadline = start + 590
    configs = [(9, 6), (6, 9), (3, 18), (2, 27)]
    best_score = -1
    best_points = None
    best_lines = None
    for idx, (c, r) in enumerate(configs):
        pts, lns = _build_grid(c, r)
        s = int(np.sum((lns[:, 0:1] * pts[:, 0:1].T + lns[:, 1:2] * pts[:, 1:2].T + lns[:, 2:3]) == 0))
        if s > best_score:
            best_score = s
            best_points = pts.copy()
            best_lines = lns.copy()
        if time.time() > deadline - 80:
            break
        time_left = deadline - 40 - (len(configs) - 1 - idx) * 15
        if time.time() < time_left:
            bp, bl, bs = _anneal(pts, lns, time_left, seed=42 + idx * 1000)
            if bs > best_score:
                best_score = bs
                best_points = bp.copy()
                best_lines = bl.copy()
    if best_points is None:
        best_points, best_lines = _build_grid(9, 6)
    if not _is_valid(best_points, best_lines):
        best_points, best_lines = _build_grid(9, 6)
    return best_points, best_lines
