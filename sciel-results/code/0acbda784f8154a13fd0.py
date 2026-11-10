import numpy as np
import time
import random

def _score(points):
    n = len(points)
    pts = np.array(points, dtype=np.int64)
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_f = i_idx[mask]
    j_f = j_idx[mask]
    k_f = k_idx[mask]
    xi = pts[i_f, 0]
    yi = pts[i_f, 1]
    xj = pts[j_f, 0]
    yj = pts[j_f, 1]
    xk = pts[k_f, 0]
    yk = pts[k_f, 1]
    area2 = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    area2 = area2[area2 > 0]
    if len(area2) == 0:
        return 0
    vals, counts = np.unique(area2, return_counts=True)
    return int(counts.max())

def _random_point():
    return (random.randint(-30, 30), random.randint(-30, 30))

def _make_grid(w, h, ox=0, oy=0):
    return [(ox + i, oy + j) for i in range(w) for j in range(h)]

def _initial_candidates():
    cands = [
        _make_grid(6, 5, -2, -2),
        _make_grid(5, 6, -2, -2),
        _make_grid(10, 3, -4, -1),
        _make_grid(15, 2, -7, -1),
        [(i - 7, 0) for i in range(15)] + [(i - 7, 2) for i in range(15)],
    ]
    random.seed(42)
    pts = set()
    while len(pts) < 30:
        pts.add(_random_point())
    cands.append(list(pts))
    return cands

def construct():
    start_time = time.time()
    deadline = start_time + 80
    best_score = 0
    best_points = None
    for cand in _initial_candidates():
        if len(cand) != 30 or len(set(cand)) != 30:
            continue
        s = _score(cand)
        if s > best_score:
            best_score = s
            best_points = list(cand)
    if best_points is None:
        best_points = _make_grid(6, 5, -2, -2)
        best_score = _score(best_points)
    random.seed(12345)
    current = list(best_points)
    current_score = best_score
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    while time.time() < deadline:
        if random.random() < 0.7:
            i = random.randint(0, 29)
            old = current[i]
            new = _random_point()
            if new == old:
                continue
            current[i] = new
            ns = _score(current)
            d = ns - current_score
            if d > 0 or random.random() < np.exp(d / max(T, 1e-10)):
                current_score = ns
                if ns > best_score:
                    best_score = ns
                    best_points = list(current)
            else:
                current[i] = old
        else:
            i, j = random.sample(range(30), 2)
            current[i], current[j] = current[j], current[i]
            ns = _score(current)
            d = ns - current_score
            if d > 0 or random.random() < np.exp(d / max(T, 1e-10)):
                current_score = ns
                if ns > best_score:
                    best_score = ns
                    best_points = list(current)
            else:
                current[i], current[j] = current[j], current[i]
        T *= cooling
        if T < T_min:
            T = T_min
    return best_points
