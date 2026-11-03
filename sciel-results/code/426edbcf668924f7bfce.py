import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 148.0
SAFETY_MARGIN = 6.0
EPS = 1e-9

def _axial_to_cartesian(axial_list):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def _build_seeds():
    # Dense hexagonal lattice of N=11 unit hexagons (touching, non-overlapping).
    # Axial ring 0..1 gives 1+6+12=19 cells; take first 11 in a compact,
    # rotationally-symmetric order to minimize the bounding hexagon.
    axial = [(0, 0),
             (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
             (2, 0), (-2, 0), (0, 2), (1, 2)]
    pts = _axial_to_cartesian(axial)
    # center the lattice
    c = pts.mean(axis=0)
    pts = pts - c
    return pts

def _smooth_S(params, pts0, cx0, cy0, temp):
    # params: [x1,y1,th1, ..., xN,yN,thN, cx, cy, beta]
    n = len(pts0)
    pos = params[:3*n].reshape(n, 3)
    cx, cy = params[3*n], params[3*n+1]
    beta = params[3*n+2]

    # rotate each inner hexagon's lattice offset by its own angle
    cos_t = np.cos(pos[:, 2])
    sin_t = np.sin(pos[:, 2])
    rx = pos[:, 0] + (pts0[:, 0]*cos_t - pts0[:, 1]*sin_t)
    ry = pos[:, 1] + (pts0[:, 0]*sin_t + pts0[:, 1]*cos_t)

    # 6 outward normals of the enclosing hexagon (apothem-direction)
    angs = beta + math.pi/6.0 + np.arange(6)*math.pi/3.0
    normals = np.stack([np.cos(angs), np.sin(angs)], axis=1)  # 6x2

    dx = rx - cx
    dy = ry - cy
    # projections: (n,6)
    projs = dx[:, None]*normals[0, None, :] + dy[:, None]*normals[1, None, :]
    allp = np.concatenate([projs.ravel(), (-projs).ravel()])

    m = allp.max()
    s = (allp - m)/temp
    s = np.clip(s, -50, 50)
    D = temp*(np.log(np.sum(np.exp(s))) + m/temp)
    S = (D + APOTHEM)/APOTHEM
    return S

def _exact_S(params, pts0):
    n = len(pts0)
    pos = params[:3*n].reshape(n, 3)
    cx, cy = params[3*n], params[3*n+1]
    beta = params[3*n+2]
    cos_t = np.cos(pos[:, 2])
    sin_t = np.sin(pos[:, 2])
    rx = pos[:, 0] + (pts0[:, 0]*cos_t - pts0[:, 1]*sin_t)
    ry = pos[:, 1] + (pts0[:, 0]*sin_t + pts0[:, 1]*cos_t)
    angs = beta + math.pi/6.0 + np.arange(6)*math.pi/3.0
    normals = np.stack([np.cos(angs), np.sin(angs)], axis=1)
    dx = rx - cx
    dy = ry - cy
    projs = dx[:, None]*normals[0, None, :] + dy[:, None]*normals[1, None, :]
    D = np.max(np.abs(projs))
    return (D + APOTHEM)/APOTHEM

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    pts0 = _build_seeds()
    n = len(pts0)
    cx0, cy0 = 0.0, 0.0

    # initial params: positions on lattice, zero rotation, center 0, beta 0
    base = np.zeros(3*n + 3)
    base[0:3*n:3] = pts0[:, 0]
    base[1:3*n:3] = pts0[:, 1]
    # angles 0, cx=0, cy=0, beta=0

    best_S = float('inf')
    best_params = base.copy()

    # Multi-start over beta and temperature annealing
    betas = [0.0, math.pi/12, math.pi/6, -math.pi/12, math.pi/6 - 1e-4]
    temps = [0.5, 0.25, 0.12, 0.06, 0.03, 0.015, 0.008, 0.004]

    for b in betas:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        cur = base.copy()
        cur[3*n+2] = b
        for temp in temps:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            try:
                res = minimize(
                    lambda p, t=temp: _smooth_S(p, pts0, cx0, cy0, t),
                    cur,
                    method='L-BFGS-B',
                    options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 20},
                )
                if res.fun < best_S and res.fun < 1e6:
                    best_S = res.fun
                    best_params = res.x.copy()
                cur = res.x.copy()
            except Exception:
                break

    # Exact polish with L-BFGS-B on the exact (nonsmooth but bounded) objective
    # using a fine log-sum-exp at tiny temp as a smooth proxy for the max.
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            res = minimize(
                lambda p: _smooth_S(p, pts0, cx0, cy0, 0.002),
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 30},
            )
            S_exact = _exact_S(res.x, pts0)
            if S_exact < best_S:
                best_S = S_exact
                best_params = res.x.copy()
    except Exception:
        pass

    # Final exact evaluation
    S_final = _exact_S(best_params, pts0)

    # Build output
    pos = best_params[:3*n].reshape(n, 3)
    cx = float(best_params[3*n])
    cy = float(best_params[3*n+1])
    beta = float(best_params[3*n+2])
    angle_deg = math.degrees(beta)

    inner = []
    for i in range(n):
        x = float(pos[i, 0])
        y = float(pos[i, 1])
        a = float(math.degrees(pos[i, 2]))
        inner.append([x, y, a])

    center = [cx, cy]
    side = float(S_final)
    rotation = angle_deg

    return inner, center, side, rotation
