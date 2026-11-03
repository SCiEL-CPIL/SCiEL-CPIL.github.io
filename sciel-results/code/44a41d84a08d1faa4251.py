import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- candidate point sets in axial coordinates ---
    candidates = []
    def add(ax):
        if len(ax) == N and len(set(ax)) == N:
            candidates.append(ax)

    # hexagonal cluster
    add([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    add([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # 3x3 block + 2
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    add([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    add([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    add([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    add([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    add([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    add([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    add([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)])
    add([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    add([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    add([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    add([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    add([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    add([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    add([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)])
    add([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)])
    add([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    add([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    add([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    add([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    add([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])
    add([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,-1),(2,-1)])
    add([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(1,-1)])
    add([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(1,-1),(2,-1)])
    add([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,-2)])

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def hex_verts(theta):
        return np.array([[math.cos(theta + k*math.pi/3.0),
                          math.sin(theta + k*math.pi/3.0)] for k in range(6)])

    def outer_normals(beta):
        return np.array([[math.cos(beta + math.pi/6.0 + k*math.pi/3.0),
                          math.sin(beta + math.pi/6.0 + k*math.pi/3.0)] for k in range(6)])

    def exact_S(pts, theta, cx, cy, beta):
        allv = []
        for i in range(N):
            allv.append(pts[i] + hex_verts(theta[i]))
        allv = np.vstack(allv)
        projs = (allv - np.array([cx, cy])) @ outer_normals(beta).T
        D = float(np.max(np.abs(projs)))
        return (D + APOTHEM) / APOTHEM

    def smooth_S(pts, theta, cx, cy, beta, temp):
        allv = []
        for i in range(N):
            allv.append(pts[i] + hex_verts(theta[i]))
        allv = np.vstack(allv)
        projs = (allv - np.array([cx, cy])) @ outer_normals(beta).T
        ap = np.abs(projs).ravel()
        m = float(np.max(ap))
        if m > 1000:
            return 1e6
        sh = (ap - m) / temp
        sh = np.clip(sh, -60, 60)
        D = m + temp * math.log(float(np.sum(np.exp(sh))))
        return (D + APOTHEM) / APOTHEM

    def pack(pts, theta0, beta0, cx0, cy0, temp):
        def obj(v):
            cx, cy, beta = v[0], v[1], v[2]
            th = v[3:]
            return smooth_S(pts, th, cx, cy, beta, temp)
        v0 = np.concatenate([[cx0, cy0, beta0], theta0])
        try:
            res = minimize(obj, v0, method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-9})
            if res.fun < 1e5:
                cx, cy, beta = res.x[0], res.x[1], res.x[2]
                th = res.x[3:]
                return exact_S(pts, th, cx, cy, beta), (cx, cy, beta, th)
        except Exception:
            pass
        return float('inf'), None

    def refine(pts, cx, cy, beta, th, temp):
        def obj(v):
            cx, cy, beta = v[0], v[1], v[2]
            th = v[3:]
            return smooth_S(pts, th, cx, cy, beta, temp)
        v0 = np.concatenate([[cx, cy, beta], th])
        try:
            res = minimize(obj, v0, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < 1e5:
                cx, cy, beta = res.x[0], res.x[1], res.x[2]
                th = res.x[3:]
                return exact_S(pts, th, cx, cy, beta), (cx, cy, beta, th)
        except Exception:
            pass
        return exact_S(pts, th, cx, cy, beta), (cx, cy, beta, th)

    best = float('inf')
    best_sol = None

    for ci, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        c0 = (float(np.mean(pts[:,0])), float(np.mean(pts[:,1])))

        for hb in range(7):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = hb * math.pi / 6.0
            theta0 = np.zeros(N)
            S, sol = pack(pts, theta0, beta, c0[0], c0[1], 0.05)
            if S < best:
                best = S
                best_sol = (pts, sol[0], sol[1], sol[2], sol[3])
            if sol is not None and S < 1e5:
                S2, sol2 = refine(pts, sol[0], sol[1], sol[2], sol[3], 0.01)
                if S2 < best:
                    best = S2
                    best_sol = (pts, sol2[0], sol2[1], sol2[2], sol2[3])

    if best_sol is None or best == float('inf'):
        pts = axial_to_cartesian(candidates[0])
        cx, cy = float(np.mean(pts[:,0])), float(np.mean(pts[:,1]))
        beta = 0.0
        theta = np.zeros(N)
        best = exact_S(pts, theta, cx, cy, beta)
        best_sol = (pts, cx, cy, beta, theta)

    pts, cx, cy, beta, theta = best_sol
    angle_deg = math.degrees(beta)

    inner = []
    for i in range(N):
        inner.append([float(pts[i,0]), float(pts[i,1]), float(math.degrees(theta[i]))])
    center = [float(cx), float(cy)]
    side = float(best)
    rotation = float(angle_deg)
    return inner, center, side, rotation
