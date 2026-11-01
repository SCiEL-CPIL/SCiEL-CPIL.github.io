import numpy as np
from scipy.optimize import minimize
import time

N = 12
START = time.time()
DEADLINE = START + 130.0  # safety margin before the 150 s wall-clock kill

def hex_vertices(cx, cy, s=1.0, angle_deg=0.0):
    a = np.radians(angle_deg)
    return np.array([(cx + s*np.cos(a + k*np.pi/3.0),
                      cy + s*np.sin(a + k*np.pi/3.0)) for k in range(6)])

def lattice_to_cart(q, r):
    # flat-topped unit-hexagon tiling: neighbor centers at (±2,0),(±1,±√3)
    return (2.0*q + r, np.sqrt(3.0)*r)

def cart_to_lattice(x, y):
    r = round(y / np.sqrt(3.0))
    q = round((x - r) / 2.0)
    return (q, r)

def enclosing_side_full(verts, maxiter=2500):
    """Minimal side of a regular hexagon (any center/rotation) containing verts."""
    cx0, cy0 = verts.mean(axis=0)
    best = None
    bestp = None
    for beta0 in [0.0, np.pi/6, np.pi/3, np.pi/2]:
        def objective(params):
            cx, cy, beta = params
            c = np.array([cx, cy])
            us = np.array([[np.cos(beta + np.pi/6 + j*np.pi/3.0),
                            np.sin(beta + np.pi/6 + j*np.pi/3.0)] for j in range(6)])
            d = (verts - c) @ us.T
            return (2.0/np.sqrt(3.0)) * d.max()
        res = minimize(objective, [cx0, cy0, beta0], method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-7, 'fatol': 1e-9})
        if best is None or res.fun < best:
            best = res.fun
            bestp = res.x
    return best, bestp

def eval_cluster(coords):
    centers = [lattice_to_cart(q, r) for q, r in coords]
    verts = np.vstack([hex_vertices(x, y) for x, y in centers])
    s, p = enclosing_side_full(verts)
    return s, centers, p

def build_initial_clusters():
    clusters = []
    clusters.append([(q, r) for r in range(2) for q in range(6)])   # 2x6
    clusters.append([(q, r) for r in range(3) for q in range(4)])   # 3x4
    clusters.append([(q, r) for r in range(4) for q in range(3)])   # 4x3
    base = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    extra = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)]
    clusters.append(base + extra)                                   # hex 7+5
    clusters.append([(q, 0) for q in range(12)])                    # 1x12
    c = []
    for r, n in enumerate([5, 4, 3]):
        for q in range(n):
            c.append((q, r))
    clusters.append(c)
    c = []
    for r, n in enumerate([3, 4, 5]):
        for q in range(n):
            c.append((q, r))
    clusters.append(c)
    clusters.append([(q, r) for r in range(4) for q in range(3)])   # 4x3b
    clusters.append([(q, r) for r in range(6) for q in range(2)])   # 6x2
    c = []
    for r, n in enumerate([4, 4, 4]):
        off = 0 if r % 2 == 0 else 1
        for q in range(off, off + n):
            c.append((q, r))
    clusters.append(c)                                              # staggered
    return clusters

def construct():
    best_s = float('inf')
    best_centers = None
    best_p = None

    for coords in build_initial_clusters():
        if time.time() > DEADLINE:
            break
        try:
            s, centers, p = eval_cluster(coords)
            if s < best_s:
                best_s = s
                best_centers = centers
                best_p = p
        except Exception:
            pass

    # Local search: relocate one cell to an empty lattice neighbor if it shrinks the hull.
    if best_centers is not None and time.time() < DEADLINE - 10:
        coords = set(cart_to_lattice(x, y) for (x, y) in best_centers)
        improved = True
        iters = 0
        while improved and iters < 30 and time.time() < DEADLINE - 10:
            improved = False
            iters += 1
            for (q, r) in sorted(coords):
                if time.time() > DEADLINE - 10:
                    break
                for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
                    nq, nr = q + dq, r + dr
                    if (nq, nr) in coords:
                        continue
                    new_coords = list(coords)
                    new_coords.remove((q, r))
                    new_coords.append((nq, nr))
                    try:
                        s, centers, p = eval_cluster(new_coords)
                    except Exception:
                        continue
                    if s < best_s - 1e-9:
                        best_s = s
                        best_centers = centers
                        best_p = p
                        coords = set(new_coords)
                        improved = True
                        break

    if best_centers is None or best_p is None:
        centers = [lattice_to_cart(q, r) for r in range(2) for q in range(6)]
        return [[float(x), float(y), 0.0] for (x, y) in centers], (0.0, 0.0), 40.0, 0.0

    inner = [[float(x), float(y), 0.0] for (x, y) in best_centers]
    cx, cy, beta = best_p
    return inner, (float(cx), float(cy)), float(best_s), float(np.degrees(beta))
