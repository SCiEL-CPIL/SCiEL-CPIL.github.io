# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import itertools

_DEADLINE = time.time() + 595  # 595s safety margin under 600s limit


def _tri_area(a, b, c):
    return abs(a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1])) / 2.0


def _all_tri_areas(pts):
    n = len(pts)
    areas = np.empty((n*(n-1)*(n-2)//6,))
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                areas[idx] = _tri_area(pts[i], pts[j], pts[k])
                idx += 1
    return areas


def _score(pts):
    """A_min / A_hull, higher is better."""
    try:
        hull = ConvexHull(pts)
        A_hull = hull.volume
        if A_hull <= 0:
            return 0.0
        A_min = _all_tri_areas(pts).min()
        if A_min <= 0:
            return 0.0
        return A_min / A_hull
    except Exception:
        return 0.0


def _negative_score(pts):
    return -_score(pts)


def _regular_polygon(n, radius=1.0, cx=0.0, cy=0.0):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + np.pi/2
    return np.column_stack([cx + radius*np.cos(angles), cy + radius*np.sin(angles)])


def _concentric_init(n=13):
    """13 points: 3 center-ish + 5 inner ring + 5 outer ring, roughly regular."""
    pts = []
    # outer ring: 5 points
    outer_r = 1.0
    a_out = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/2
    pts.extend(list(outer_r * np.column_stack([np.cos(a_out), np.sin(a_out)])))
    # middle ring: 5 points, rotated
    mid_r = 0.55
    a_mid = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/2 + np.pi/5
    pts.extend(list(mid_r * np.column_stack([np.cos(a_mid), np.sin(a_mid)])))
    # inner: 3 points in a small triangle
    inner_r = 0.2
    a_in = np.linspace(0, 2*np.pi, 3, endpoint=False) + np.pi/3
    pts.extend(list(inner_r * np.column_stack([np.cos(a_in), np.sin(a_in)])))
    return np.array(pts)


def _optimize_from(pts0, time_budget=120.0):
    """Local optimization: repeatedly move the worst-triangle vertices."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_s = _score(pts)
    t0 = time.time()

    # Phase 1: coordinate-wise Nelder-Mead on all 26 coords
    def obj(x):
        p = x.reshape(-1, 2)
        try:
            ConvexHull(p)
        except Exception:
            return 1e6
        a_min = _all_tri_areas(p).min()
        if a_min < 1e-12:
            return 1e6
        try:
            A_hull = ConvexHull(p).volume
        except Exception:
            return 1e6
        return -a_min / A_hull

    res = minimize(obj, pts.ravel(), method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-12,
                            'adaptive': True})
    if res.fun < obj(pts.ravel()):
        cand = res.x.reshape(-1, 2)
        s = _score(cand)
        if s > best_s:
            best_s, best_pts = s, cand.copy()
            pts = cand.copy()

    if time.time() - t0 > time_budget:
        return best_pts

    # Phase 2: targeted moves on min-area triangle vertices
    for it in range(400):
        if time.time() - t0 > time_budget:
            break
        areas = _all_tri_areas(pts)
        mi = np.argmin(areas)
        # map linear index back to (i,j,k)
        n = len(pts)
        ii, jj, kk = 0, 0, 0
        idx = 0
        found = False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if idx == mi:
                        ii, jj, kk = i, j, k
                        found = True
                        break
                    idx += 1
                if found: break
            if found: break

        # Try moving each of the 3 vertices a small step toward centroid of others
        centroid = (pts[ii] + pts[jj] + pts[kk]) / 3.0
        for v in [ii, jj, kk]:
            for step in [0.02, 0.05, 0.1]:
                trial = pts.copy()
                trial[v] = trial[v] + step * (centroid - trial[v])
                s = _score(trial)
                if s > best_s:
                    best_s, best_pts = s, trial.copy()
                    pts = trial.copy()
                    break

    return best_pts


def _multi_start():
    t_start = time.time()
    candidates = []
    # 1) regular 13-gon
    candidates.append(_regular_polygon(13, radius=1.0))
    # 2) concentric 5-5-3
    candidates.append(_concentric_init(13))
    # 3) regular 12-gon + center
    p12 = _regular_polygon(12, radius=1.0)
    candidates.append(np.vstack([p12, np.array([[0.0, 0.0]])]))
    # 4) regular 6 outer + 6 inner + 1 center
    p6o = _regular_polygon(6, radius=1.0)
    p6i = _regular_polygon(6, radius=0.5) + 0.5*np.cos(np.pi/6)
    candidates.append(np.vstack([p6o, p6i, np.array([[0.0, 0.0]])]))

    best_pts = candidates[0]
    best_s = _score(best_pts)

    for c in candidates:
        if time.time() - t_start > 560:
            break
        s = _score(c)
        if s > best_s:
            best_s, best_pts = s, c.copy()

    # optimize the best initial
    result = _optimize_from(best_pts, time_budget=200.0)
    s2 = _score(result)
    if s2 > best_s:
        best_pts, best_s = result, s2

    # try optimizing a couple of other inits if time remains
    for c in candidates:
        if time.time() - t_start > 575:
            break
        if np.allclose(c, best_pts):
            continue
        r = _optimize_from(c, time_budget=60.0)
        s = _score(r)
        if s > best_s:
            best_pts, best_s = r, s

    return best_pts


def heilbronn_convex13() -> np.ndarray:
    pts = _multi_start()
    # Final sanity
    if pts.shape != (13, 2) or not np.all(np.isfinite(pts)):
        return _regular_polygon(13, radius=1.0)
    return pts


# EVOLVE-BLOCK-END
