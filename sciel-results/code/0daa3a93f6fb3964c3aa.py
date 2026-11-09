# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _normalize(pts):
    centroid = pts.mean(axis=0)
    p = pts - centroid
    d = np.linalg.norm(p, axis=1)
    mean_d = d.mean()
    if mean_d <= 0.01:
        return None, None, None
    p = p / mean_d
    return p, d, mean_d

def _check_strictly_convex(p):
    n = len(p)
    if n < 3:
        return False
    # no repeated points
    for i in range(n):
        if np.linalg.norm(p[i] - p[(i + 1) % n]) < 1e-9:
            return False
    # compute cross products of consecutive edges
    signs = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            return False
        signs.append(1 if cross > 0 else -1)
    # all same sign
    if signs.count(signs[0]) != n:
        return False
    # total turning = +-2*pi
    angles = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_a = np.clip(cos_a, -1, 1)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sin_a = cross / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ang = np.arctan2(sin_a, cos_a)
        angles.append(ang)
    total_turn = sum(angles)
    expected = 2 * np.pi if signs[0] > 0 else -2 * np.pi
    if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
        return False
    return True

def _vertex_score(p, v_idx):
    n = len(p)
    v = p[v_idx]
    # distances to all other vertices
    d = np.array([np.linalg.norm(v - p[j]) for j in range(n) if j != v_idx])
    d_sorted = np.sort(d)
    if np.any(d_sorted < 0.05):
        return 0.0
    # find k: first index minimizing d_{k+1} - d_k for k in 0..17
    gaps = d_sorted[1:] - d_sorted[:-1]
    k = int(np.argmin(gaps))  # 0..17
    D = (d_sorted[k] + d_sorted[k + 1]) / 2
    if D < 0.1:
        return 0.0
    # select four distances
    if k == 0:
        selected = d_sorted[0:4]
    elif k == 17:
        selected = d_sorted[15:19]
    else:
        selected = d_sorted[k - 1:k + 3]
    if len(selected) < 4:
        return 0.0
    raw = np.mean([max(di / D, D / di) for di in selected])
    if raw < 1e-12:
        return 0.0
    score = min(1.0, 1.0 / raw)
    return score

def _score_polygon(p):
    """Score a polygon. Returns min vertex score (0 if invalid)."""
    pn, _, _ = _normalize(p)
    if pn is None:
        return 0.0
    if not _check_strictly_convex(pn):
        return 0.0
    min_vs = 1.0
    for i in range(len(pn)):
        vs = _vertex_score(pn, i)
        if vs < min_vs:
            min_vs = vs
        if min_vs < 0.01:  # early exit
            break
    return min_vs

def _regular_polygon(n=20, radius=1.0):
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600.0  # 600s deadline, leave 20s margin

    n = 20
    # Start with the regular 20-gon
    pts = _regular_polygon(n, 1.0)
    best_score = _score_polygon(pts)
    best_pts = pts.copy()

    if best_score <= 0.0:
        # Fallback: return regular polygon (shouldn't happen)
        return pts

    # Use Nelder-Mead to maximize the minimum vertex score by minimizing its negative
    # Work in 40D: flatten the (20,2) array
    x0 = best_pts.flatten()

    def objective(x):
        try:
            if time.time() > deadline:
                return 1000.0
            p = x.reshape((n, 2))
            score = _score_polygon(p)
            if score == 0.0:
                return 1000.0
            return -score
        except Exception:
            return 1000.0

    # Run Nelder-Mead with tight tolerances
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'maxfev': 5000,
                'xatol': 1e-8,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        if result.fun < 1000.0:
            new_score = -result.fun
            if new_score > best_score:
                best_score = new_score
                best_pts = result.x.reshape((n, 2))
    except Exception:
        pass

    # If we have time, do a second, finer pass
    if time.time() < deadline - 30:
        try:
            result2 = minimize(
                objective,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'maxfev': 3000,
                    'xatol': 1e-10,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if result2.fun < 1000.0:
                new_score2 = -result2.fun
                if new_score2 > best_score:
                    best_score = new_score2
                    best_pts = result2.x.reshape((n, 2))
        except Exception:
            pass

    return best_pts

# EVOLVE-BLOCK-END
