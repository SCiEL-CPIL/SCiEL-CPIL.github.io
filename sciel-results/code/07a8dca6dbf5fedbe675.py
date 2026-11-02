import math
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
S3 = math.sqrt(3.0)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY = 12.0

def _hex_vertices(cx, cy, ang_deg):
    a = math.radians(ang_deg)
    c, s = math.cos(a), math.sin(a)
    out = []
    for k in range(6):
        t = k * math.pi / 3.0
        vx, vy = math.cos(t), math.sin(t)
        out.append((cx + vx * c - vy * s, cy + vx * s + vy * c))
    return out

def _poly_overlap_area(a, b):
    # Sutherland-Hodgman of a (subject) clipped by b (clip), returns area
    def inside(p, e1, e2):
        return (e2[0] - e1[0]) * (p[1] - e1[1]) - (e2[1] - e1[1]) * (p[0] - e1[0]) <= 1e-12
    def intersect(s1, s2, c1, c2):
        x1, y1 = s1; x2, y2 = s2; x3, y3 = c1; x4, y4 = c2
        d1x, d1y = x2 - x1, y2 - y1
        d2x, d2y = x4 - x3, y4 - y3
        den = d1x * d2y - d1y * d2x
        if abs(den) < 1e-15:
            return s1
        t = ((x3 - x1) * d2y - (y3 - y1) * d2x) / den
        return (x1 + t * d1x, y1 + t * d1y)
    out = list(a)
    for i in range(len(b)):
        if not out:
            break
        e1, e2 = b[i], b[(i + 1) % len(b)]
        inp = []
        for j in range(len(out)):
            c = out[j]
            n = out[(j + 1) % len(out)]
            ci = inside(c, e1, e2)
            ni = inside(n, e1, e2)
            if ci:
                inp.append(c)
                if not ni:
                    inp.append(intersect(c, n, e1, e2))
            elif ni:
                inp.append(intersect(c, n, e1, e2))
        out = inp
    if len(out) < 3:
        return 0.0
    area = 0.0
    for j in range(len(out)):
        x1, y1 = out[j]
        x2, y2 = out[(j + 1) % len(out)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5

def _pairwise_overlap_area(V):
    # V: (N,6,2)
    n = V.shape[0]
    total = 0.0
    for i in range(n):
        hi = _hex_vertices(V[i, 0, 0], V[i, 0, 1], V[i, 2])
        for j in range(i + 1, n):
            hj = _hex_vertices(V[j, 0, 0], V[j, 0, 1], V[j, 2])
            a = _poly_overlap_area(hi, hj)
            if a > 1e-12:
                total += a
    return total

def _outer_side(V):
    # minimal enclosing regular hexagon side (rotation free) via LP over 6 normals
    best = float('inf')
    P = V[:, :2]
    for k in range(6):
        beta = k * math.pi / 6.0
        normals = []
        for m in range(6):
            ang = beta + math.pi / 6.0 + m * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        A_ub = []
        b_ub = []
        for nx, ny in normals:
            for px, py in P:
                A_ub.append([-nx, -ny, -1.0]); b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0]);  b_ub.append(px * nx + py * ny)
        A_ub = np.array(A_ub); b_ub = np.array(b_ub)
        try:
            res = linprog(np.array([0.0, 0.0, 1.0]), A_ub=A_ub, b_ub=b_ub,
                          bounds=[(None, None), (None, None), (0, None)], method='highs')
            if res.status == 0:
                D = res.x[2]
            else:
                cx, cy = P[:, 0].mean(), P[:, 1].mean()
                D = 0.0
                for nx, ny in normals:
                    proj = np.abs((P[:, 0] - cx) * nx + (P[:, 1] - cy) * ny)
                    D = max(D, proj.max())
        except Exception:
            cx, cy = P[:, 0].mean(), P[:, 1].mean()
            D = 0.0
            for nx, ny in normals:
                proj = np.abs((P[:, 0] - cx) * nx + (P[:, 1] - cy) * ny)
                D = max(D, proj.max())
        S = (D + APOTHEM) / APOTHEM
        if S < best:
            best = S
    return best

def _objective(params, S0, mu):
    if time.time() - START_TIME > TIME_LIMIT - SAFETY:
        return float('inf')
    x = params[:2 * N]
    ang = params[2 * N:3 * N]
    V = np.zeros((N, 6, 2))
    for i in range(N):
        cx, cy = x[2 * i], x[2 * i + 1]
        a = math.radians(ang[i])
        c, s = math.cos(a), math.sin(a)
        for k in range(6):
            t = k * math.pi / 3.0
            vx, vy = math.cos(t), math.sin(t)
            V[i, k, 0] = cx + vx * c - vy * s
            V[i, k, 1] = cy + vx * s + vy * c
    S = _outer_side(V)
    ov = _pairwise_overlap_area(V)
    return S + mu * ov

def _valid(V):
    ov = _pairwise_overlap_area(V)
    return ov < 1e-7

def _build(V):
    cx = float(V[:, 0, 0].mean()); cy = float(V[:, 0, 1].mean())
    # outer hexagon: rotation 0, centered at centroid of centers, side = enclosing side
    S = _outer_side(V)
    inner = []
    for i in range(N):
        inner.append([float(V[i, 0, 0]), float(V[i, 0, 1]), float(V[i, 2])])
    return inner, [cx, cy], float(S), 0.0

def _seed_axial():
    axial = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
             (1, 1), (-1, 2), (-2, 1), (2, -1)]
    V = np.zeros((N, 6, 2))
    for i, (q, r) in enumerate(axial):
        x = S3 * (q + r / 2.0)
        y = 1.5 * r
        for k in range(6):
            t = k * math.pi / 3.0
            V[i, k, 0] = x + math.cos(t)
            V[i, k, 1] = y + math.sin(t)
    return V

def _seed_cluster():
    # 1 center + 6 ring (touching) + 4 in the 4 wide gaps
    V = np.zeros((N, 6, 2))
    centers = [(0.0, 0.0)]
    for k in range(6):
        t = k * math.pi / 3.0
        centers.append((2.0 * math.cos(t), 2.0 * math.sin(t)))
    # 4 gap centers at radius ~3.464 (2*sqrt(3)) at 45,135,225,315
    R = 2.0 * math.sqrt(3.0)
    for k in range(4):
        t = math.pi / 4.0 + k * math.pi / 2.0
        centers.append((R * math.cos(t), R * math.sin(t)))
    for i in range(N):
        cx, cy = centers[i]
        for k in range(6):
            t = k * math.pi / 3.0
            V[i, k, 0] = cx + math.cos(t)
            V[i, k, 1] = cy + math.sin(t)
    return V

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- deterministic seeds ---
    V_axial = _seed_axial()
    V_cluster = _seed_cluster()
    best_V = None
    best_S = float('inf')
    for V in (V_axial, V_cluster):
        if _valid(V):
            S = _outer_side(V)
            if S < best_S:
                best_S = S
                best_V = V.copy()
    if best_V is None:
        best_V = V_axial
        best_S = _outer_side(V_axial)

    S0 = best_S
    rng = np.random.default_rng(7)

    # --- direct continuous optimization, time-budgeted ---
    mu = 0.1
    deadline = START_TIME + TIME_LIMIT - SAFETY
    while time.time() < deadline:
        # randomize around best
        V0 = best_V.copy()
        V0[:, 0, 0] += rng.normal(0, 0.2, N)
        V0[:, 0, 1] += rng.normal(0, 0.2, N)
        V0[:, 2] += rng.uniform(-0.2, 0.2, N)
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[2 * i] = V0[i, 0, 0]
            x0[2 * i + 1] = V0[i, 0, 1]
            x0[2 * N + i] = V0[i, 2]

        try:
            res = minimize(_objective, x0, args=(S0, mu), method='L-BFGS-B',
                           options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-4})
        except Exception:
            res = None

        if res is not None and res.fun < 1e8:
            xf = res.x
            Vf = np.zeros((N, 6, 2))
            for i in range(N):
                cx, cy = xf[2 * i], xf[2 * i + 1]
                a = math.radians(xf[2 * N + i])
                c, s = math.cos(a), math.sin(a)
                for k in range(6):
                    t = k * math.pi / 3.0
                    vx, vy = math.cos(t), math.sin(t)
                    Vf[i, k, 0] = cx + vx * c - vy * s
                    Vf[i, k, 1] = cy + vx * s + vy * c
            if _valid(Vf):
                S = _outer_side(Vf)
                if S < best_S:
                    best_S = S
                    best_V = Vf.copy()
                    S0 = S
        # anneal penalty for later passes
        mu *= 0.6
        if mu < 1e-3:
            mu = 0.05

    if best_V is None:
        best_V = V_axial
    return _build(best_V)
