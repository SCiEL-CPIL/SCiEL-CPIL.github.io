"""Heilbronn-11: kinetic annealing with topological collision resolution + barycentric DE."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize, differential_evolution

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4
DEADLINE = 142

# Precompute triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])


def _all_areas(pts):
    pi = pts[_T_I]; pj = pts[_T_J]; pk = pts[_T_K]
    cross = (pj[:,0]-pi[:,0])*(pk[:,1]-pi[:,1]) - (pj[:,1]-pi[:,1])*(pk[:,0]-pi[:,0])
    return np.abs(cross)*0.5


def _min_area(pts):
    return np.min(_all_areas(pts))


def _soft_min(pts, beta):
    areas = _all_areas(pts)
    m = np.max(areas)
    return m - np.log(np.sum(np.exp(-beta*(areas-m))))/beta


def _project(pts):
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x = max(y/S3, min(1-y/S3, x))
        pts[i] = [x, y]
    return pts


def _cart_to_bary(pts):
    c = pts[:,1]/S3H
    b = pts[:,0] - 0.5*c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])


def _bary_to_cart(abc):
    pts = np.zeros((len(abc), 2))
    pts[:,0] = abc[:,1] + 0.5*abc[:,2]
    pts[:,1] = abc[:,2]*S3H
    return pts


def _bary_pen(abc):
    pen = 0.0
    for i in range(len(abc)):
        for j in range(3):
            if abc[i,j] < 0: pen += (abc[i,j])**2 * 100
            elif abc[i,j] > 1: pen += (abc[i,j]-1)**2 * 100
    pen += np.sum((abc.sum(axis=1)-1.0)**2)*100
    return pen


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)
    best_pts = None
    best_area = 0.0

    def _ok(margin):
        return (time.time()-t0) < (DEADLINE - margin)

    def _upd(pts):
        nonlocal best_pts, best_area
        pts = _project(pts.copy())
        a = _min_area(pts)
        if a > best_area:
            best_area = a
            best_pts = pts.copy()

    # Symmetric initial config
    init = np.array([
        [0.0,0.0],[1.0,0.0],[0.5,S3H],
        [0.5,0.0],[0.75,S3H/2],[0.25,S3H/2],
        [1/3,S3H/3],[0.333,0.0],[0.667,0.0],
        [0.5,S3H/3],[0.2,0.1]
    ], dtype=float)
    _upd(init)

    # ---- Phase A: Differential Evolution in barycentric (b,c) space ----
    if _ok(55):
        def _de_obj(flat):
            bc = flat.reshape(-1, 2)
            b = bc[:,0]; c = bc[:,1]; a = 1.0-b-c
            abc = np.column_stack([a,b,c])
            pen = 0.0
            for i in range(N):
                for j in range(3):
                    v = abc[i,j]
                    if v < 0: pen += v*v*500
                    elif v > 1: pen += (v-1)**2*500
            pts = _bary_to_cart(abc)
            return -_min_area(pts) + pen

        bounds = [(0,1),(0,1)]*N
        try:
            de_res = differential_evolution(_de_obj, bounds, maxiter=30, popsize=12,
                                             seed=42, tol=1e-4, polish=False, init='latinhypercube')
            bc = de_res.x.reshape(-1,2)
            pts = _bary_to_cart(np.column_stack([1-bc[:,0]-bc[:,1], bc[:,0], bc[:,1]]))
            _upd(pts)
        except Exception:
            pass

    # ---- Phase B: Kinetic Annealing with Collision Resolution ----
    if best_pts is not None and _ok(40):
        pts = best_pts.copy()
        # Annealing schedule
        n_iter = 300
        for it in range(n_iter):
            if not _ok(8):
                break
            T = 0.05 * (1 - it/n_iter) + 0.001
            # Find minimum-area triangle
            areas = _all_areas(pts)
            idx = np.argmin(areas)
            i, j, k = _TRIPLETS[idx]
            pi, pj, pk = pts[i], pts[j], pts[k]
            # Push apart the three vertices of the min triangle
            for a, b in [(i,j),(j,k),(i,k)]:
                d = pts[a] - pts[b]
                dist = np.linalg.norm(d)
                if dist > 1e-12:
                    force = T / (dist + 0.01)
                    pts[a] += d/dist * force
                    pts[b] -= d/dist * force
            # Also apply gentle repulsion from all other points
            for a in range(N):
                for b in range(a+1, N):
                    d = pts[a] - pts[b]
                    dist = np.linalg.norm(d)
                    if dist > 1e-12 and dist < 0.3:
                        force = T * 0.1 / (dist + 0.05)
                        pts[a] += d/dist * force
                        pts[b] -= d/dist * force
            _project(pts)
            _upd(pts)

    # ---- Phase C: Multi-start Nelder-Mead polish ----
    if best_pts is not None and _ok(25):
        def _nm_obj(x):
            p = _project(x.reshape(-1,2).copy())
            return -_min_area(p)

        # Generate diverse starts
        starts = [best_pts.copy()]
        for _ in range(8):
            cand = best_pts + rng.normal(0, 0.03, size=(N,2))
            _project(cand)
            starts.append(cand)

        for s in starts:
            if not _ok(5):
                break
            try:
                res = minimize(_nm_obj, s.flatten(), method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
                p = _project(res.x.reshape(-1,2).copy())
                _upd(p)
            except Exception:
                pass

    # ---- Phase D: Final fine polish ----
    if best_pts is not None and _ok(5):
        def _nm_obj2(x):
            p = _project(x.reshape(-1,2).copy())
            return -_min_area(p)
        try:
            res = minimize(_nm_obj2, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 6000, 'xatol': 1e-12, 'fatol': 1e-14})
            p = _project(res.x.reshape(-1,2).copy())
            _upd(p)
        except Exception:
            pass

    if best_pts is None:
        pts = np.zeros((N,2))
        for i in range(N):
            u, v = rng.random(), rng.random()
            if u+v > 1: u, v = 1-u, 1-v
            pts[i] = [u + 0.5*v, S3H*v]
        return pts

    _project(best_pts)
    return best_pts
