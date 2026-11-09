"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2.0
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) = 165 triangle index triplets
_i, _j, _k = [], [], []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            _i.append(a); _j.append(b); _k.append(c)
_i = np.array(_i)
_j = np.array(_j)
_k = np.array(_k)
_M = len(_i)  # 165

DEADLINE = 115.0  # safety margin before 120s kill


def _min_area(points):
    """Minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_i]
    p2 = points[_j]
    p3 = points[_k]
    # area = 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.min(np.abs(cross))


def _inside_penalty(points):
    """Penalty for points outside the triangle (should be ~0)."""
    # Barycentric-ish: for triangle A(0,0), B(1,0), C(0.5,H)
    # point (x,y) is inside iff y >= 0, y <= H, and x between the two side lines.
    # Side AB: y=0 -> y>=0
    # Side BC: line from (1,0) to (0.5,H): slope = H/(0.5-1)= -2H; y - 0 = -2H (x - 1) -> y = -2H x + 2H
    #   inside: y <= -2H x + 2H  i.e. 2H x + y <= 2H
    # Side CA: line from (0.5,H) to (0,0): slope = H/0.5 = 2H; y = 2H x; inside: y >= 2H x  i.e. 2H x - y <= 0
    x = points[:, 0]
    y = points[:, 1]
    pen = 0.0
    pen += np.sum(np.maximum(0.0, -y))            # y < 0
    pen += np.sum(np.maximum(0.0, y - H))          # y > H
    pen += np.sum(np.maximum(0.0, 2.0 * H * x + y - 2.0 * H))  # right side
    pen += np.sum(np.maximum(0.0, 2.0 * H * x - y))            # left side
    return pen


def _objective(flat):
    points = flat.reshape(N, 2)
    val = _min_area(points)
    pen = _inside_penalty(points)
    # maximize val, so minimize -(val) + big penalty
    return -(val) + 100.0 * pen


def construct():
    t0 = time.time()
    best_flat = None
    best_val = -np.inf

    rng = np.random.default_rng(7)

    # Seed with a few good initial configurations
    def _rand_inside():
        # random point in triangle via barycentric
        u = rng.random()
        v = rng.random()
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        # barycentric (u, v, 1-u-v) wrt A,B,C
        x = u * B[0] + v * C[0]
        y = u * B[1] + v * C[1]
        return x, y

    def _make_start(mode):
        pts = np.zeros((N, 2))
        if mode == 0:
            # grid-ish: spread over triangle
            for idx in range(N):
                pts[idx] = _rand_inside()
        elif mode == 1:
            # vertices + centroid + ring
            pts[0] = A + 0.05 * (rng.random(2) - 0.5)
            pts[1] = B + 0.05 * (rng.random(2) - 0.5)
            pts[2] = C + 0.05 * (rng.random(2) - 0.5)
            for idx in range(3, N):
                pts[idx] = _rand_inside()
        else:
            for idx in range(N):
                pts[idx] = _rand_inside()
        return pts

    def _try_start(pts):
        nonlocal best_flat, best_val
        if time.time() > DEADLINE:
            return
        flat = pts.reshape(-1).copy()
        try:
            res = minimize(_objective, flat, method='L-BFGS-B',
                           options={'maxiter': 600, 'maxls': 50})
            if res.success or res.fun < _objective(flat):
                cand = res.x
                val = _min_area(cand.reshape(N, 2))
                if _inside_penalty(cand.reshape(N, 2)) < 1e-6 and val > best_val:
                    best_val = val
                    best_flat = cand.copy()
        except Exception:
            pass

    # Multiple starts with increasing effort
    n_starts = 0
    for mode in range(3):
        for trial in range(8):
            if time.time() > DEADLINE:
                break
            pts = _make_start(mode)
            _try_start(pts)
            n_starts += 1
        if time.time() > DEADLINE:
            break

    # Additional pure-random starts if time remains
    while time.time() < DEADLINE - 5.0:
        pts = _make_start(2)
        _try_start(pts)
        n_starts += 1

    # If we never found a valid point, fall back to a simple valid configuration
    if best_flat is None:
        pts = np.zeros((N, 2))
        for idx in range(N):
            pts[idx] = _rand_inside()
        best_flat = pts.reshape(-1)

    return best_flat.reshape(N, 2)
