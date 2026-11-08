"""Rewritten program: unit hexagons in hexagon n=12, minimize outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

def _hex_verts(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _polygon_overlap_depth(A, B):
    """SAT overlap depth between convex polygons A and B (arrays of 2D verts). 0 if disjoint."""
    min_depth = float('inf')
    axes = []
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            length = np.linalg.norm(edge)
            if length > 1e-12:
                axes.append(np.array([-edge[1], edge[0]]) / length)
    for ax in axes:
        pa = np.dot(A, ax)
        pb = np.dot(B, ax)
        if pa.max() <= pb.min() or pb.max() <= pa.min():
            return 0.0
        ov = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
        if ov < min_depth:
            min_depth = ov
    return min_depth

def _containment_violation(inner_verts, outer_verts):
    """Sum of distances by which inner verts fall outside outer (edge-wise)."""
    n_out = len(outer_verts)
    pen = 0.0
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            el = np.linalg.norm(edge)
            if el < 1e-12:
                continue
            nrm = np.array([-edge[1], edge[0]]) / el
            # For CCW outer, interior is to the left (positive cross of edge with (p-v1))
            # signed dist = dot(p - v1, nrm)
            sd = np.dot(iv - v1, nrm)
            if sd < 0:
                pen += -sd
    return pen

def _objective(params, S, center, outer_angle):
    # params: 3*N floats (x,y,angle_deg per hex)
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    pen = 0.0
    for iv in inner_list:
        pen += _containment_violation(iv, outer)
    for i in range(N):
        for j in range(i+1, N):
            d = _polygon_overlap_depth(inner_list[i], inner_list[j])
            if d > 0:
                pen += d
    return pen

def _make_grid(nrows, ncols):
    pos = []
    for r in range(nrows):
        for c in range(ncols):
            x = c * 2.0 + (1.0 if r % 2 else 0.0)
            y = r * np.sqrt(3.0)
            pos.append([x, y])
    arr = np.array(pos)
    arr -= arr.mean(axis=0)
    return arr

def _ring_init(radius=2.0):
    pts = []
    for i in range(6):
        a = np.radians(60*i)
        pts.append([radius*np.cos(a), radius*np.sin(a)])
    for i in range(6):
        a = np.radians(60*i + 30)
        pts.append([(radius*1.5)*np.cos(a), (radius*1.5)*np.sin(a)])
    return np.array(pts) - np.array(pts).mean(axis=0)

def _linear_init(S):
    # spread along a line
    xs = np.linspace(-S*0.7, S*0.7, N)
    return np.column_stack([xs, np.zeros(N)])

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin

    best = None  # (S, params, center, outer_angle)
    best_S = float('inf')

    centers = [(0.0, 0.0)]
    outer_angles = [0.0, 30.0]

    inits = []
    # hex grids
    for nr, nc in [(2, 6), (3, 4), (4, 3), (6, 2)]:
        if nr * nc == N:
            inits.append(_make_grid(nr, nc))
    # ring
    inits.append(_ring_init(2.0))
    inits.append(_ring_init(2.5))
    # linear
    inits.append(_linear_init(4.0))
    inits.append(_linear_init(3.5))
    # random
    rng = np.random.default_rng(42)
    for _ in range(3):
        r = rng.uniform(0, 2.5, size=(N, 2))
        inits.append(r - r.mean(axis=0))

    S_candidates = [5.0, 4.5, 4.2, 4.0, 3.8, 3.6, 3.5, 3.4, 3.3]

    for S in S_candidates:
        if time.time() > DEADLINE - 20:
            break
        for oa in outer_angles:
            if time.time() > DEADLINE - 10:
                break
            for ctr in centers:
                if time.time() > DEADLINE - 10:
                    break
                for init_pts in inits:
                    if time.time() > DEADLINE - 5:
                        break
                    init_params = np.zeros(3 * N)
                    for i in range(N):
                        init_params[3*i] = init_pts[i, 0]
                        init_params[3*i+1] = init_pts[i, 1]
                        init_params[3*i+2] = 0.0
                    def obj(p, S=S, ctr=ctr, oa=oa):
                        return _objective(p, S, ctr, oa)
                    try:
                        res = minimize(obj, init_params, method='L-BFGS-B',
                                       options={'maxiter': 300, 'maxfun': 8000,
                                                'ftol': 1e-14, 'gtol': 1e-9})
                    except Exception:
                        continue
                    if res.fun < 1e-6:
                        # valid packing for S
                        if S < best_S:
                            best_S = S
                            best = (S, res.x, ctr, oa)
                        # try to go smaller immediately
                        break
                if time.time() > DEADLINE - 10:
                    break
            if time.time() > DEADLINE - 10:
                break

    if best is None:
        # Fallback: a safe grid in a big hexagon
        grid = _make_grid(3, 4)
        inner = [[float(grid[i,0]), float(grid[i,1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 5.0, 0.0

    S, params, ctr, oa = best
    inner = [[float(params[3*i]), float(params[3*i+1]), float(params[3*i+2])] for i in range(N)]
    return inner, (float(ctr[0]), float(ctr[1])), float(S), float(oa)

# EVOLVE-BLOCK-END
