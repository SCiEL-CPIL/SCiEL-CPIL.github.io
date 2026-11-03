import math
import time
import numpy as np

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # ---------- geometry helpers ----------
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Build a good discrete starting layout (a "flower" of 7 + 4 extra).
    axial_seed = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2),
    ]
    seed_pts = np.array([axial_to_cartesian(q, r) for (q, r) in axial_seed])

    def eval_soft(pts, beta, cx, cy, temp):
        """Return (S, D, g_pts, g_beta, g_cx, g_cy) using smooth log-sum-exp surrogate."""
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])  # (6,2)
        P = pts - np.array([cx, cy])          # (n,2)
        proj = P @ normals.T                  # (n,6)
        allp = np.concatenate([proj.ravel(), (-proj).ravel()])  # (12n,)
        m = np.max(allp)
        if m > 1e6:
            return 1e12, m, np.zeros_like(pts), 0.0, 0.0, 0.0
        z = np.exp(np.clip((allp - m) / temp, -60.0, 60.0))
        Z = np.sum(z)
        w = z / Z
        # split weights back into proj (n,6) and -proj (n,6)
        wp = w[:proj.size].reshape(proj.shape)
        wn = w[proj.size:].reshape(proj.shape)
        D = temp * (math.log(Z) + m / temp)
        S = (D + APOTHEM) / APOTHEM
        # dD/dproj = temp * (wp - wn)
        dD_dproj = temp * (wp - wn)          # (n,6)
        # dD/dpts = dD_dproj @ normals
        g_pts = dD_dproj @ normals           # (n,2)
        # dD/dbeta: dproj/dbeta = -P @ T where T is (6,2) tangent
        T = np.array([
            (-math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])  # (6,2)
        dproj_dbeta = -P @ T                 # (n,6)
        g_beta = np.sum(dD_dproj * dproj_dbeta)
        # dD/dc = -dD/dpts summed
        g_cx = -np.sum(g_pts[:, 0])
        g_cy = -np.sum(g_pts[:, 1])
        return S, D, g_pts, g_beta, g_cx, g_cy

    def eval_exact(pts, beta, cx, cy):
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        P = pts - np.array([cx, cy])
        proj = P @ normals.T
        D = float(np.max(np.abs(proj)))
        S = (D + APOTHEM) / APOTHEM
        return S, D

    # ---------- gradient descent on continuous layout ----------
    def optimize(pts0, beta0, cx0, cy0):
        pts = pts0.copy()
        beta = beta0
        cx = cx0
        cy = cy0
        temps = [0.35, 0.2, 0.1, 0.05, 0.02, 0.008, 0.003]
        lr0 = 0.02
        for ti, temp in enumerate(temps):
            lr = lr0 * (0.6 ** ti)
            iters = 400
            for it in range(iters):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                S, D, g_pts, g_beta, g_cx, g_cy = eval_soft(pts, beta, cx, cy, temp)
                if not np.isfinite(S):
                    break
                # step
                pts = pts - lr * g_pts
                beta = beta - lr * g_beta
                cx = cx - lr * g_cx
                cy = cy - lr * g_cy
                # keep beta in [0, pi/6)
                beta = ((beta % (math.pi / 6.0)) + math.pi / 6.0) % (math.pi / 6.0)
            # re-center to reduce drift
            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        return pts, beta, cx, cy

    # ---------- refine outer hexagon for a fixed layout (exact) ----------
    def refine_outer(pts):
        best_S, best_beta, best_cx, best_cy = eval_exact(pts, 0.0,
                                                          float(np.mean(pts[:, 0])),
                                                          float(np.mean(pts[:, 1])))
        # coarse grid over beta
        for i in range(13):
            b = i * (math.pi / 6.0) / 12.0
            S, _ = eval_exact(pts, b, float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
            if S < best_S:
                best_S, best_beta = S, b
        # local polish via Nelder-Mead on exact objective
        try:
            from scipy.optimize import minimize
            def obj(v):
                b = ((v[2] % (math.pi / 6.0)) + math.pi / 6.0) % (math.pi / 6.0)
                S, _ = eval_exact(pts, b, v[0], v[1])
                return S
            for _ in range(3):
                init = np.array([best_cx, best_cy, best_beta])
                res = minimize(obj, init, method='Nelder-Mead',
                               options={'maxiter': 400, 'xatol': 1e-12, 'fatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_cx, best_cy = float(res.x[0]), float(res.x[1])
                    best_beta = ((res.x[2] % (math.pi / 6.0)) + math.pi / 6.0) % (math.pi / 6.0)
        except Exception:
            pass
        return best_S, best_beta, best_cx, best_cy

    # ---------- main search: several perturbed starts, keep best ----------
    best_S = float('inf')
    best_pts = None
    best_beta = 0.0
    best_cx = 0.0
    best_cy = 0.0

    rng = np.random.default_rng(12345)

    starts = []
    # start 1: the seed layout
    starts.append(seed_pts.copy())
    # starts 2..: seed + random jitter, and a couple of alternative discrete layouts
    for k in range(4):
        starts.append(seed_pts + rng.normal(0, 0.08, size=seed_pts.shape))

    # a few alternative discrete compact layouts
    alt_layouts = [
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(0,3),(1,3)],
    ]
    for ax in alt_layouts:
        starts.append(np.array([axial_to_cartesian(q, r) for (q, r) in ax]))

    for s in starts:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        cx0, cy0 = float(np.mean(s[:, 0])), float(np.mean(s[:, 1]))
        # quick outer refine on the raw start to seed beta
        S0, b0, cx0, cy0 = refine_outer(s)
        pts, beta, cx, cy = optimize(s, b0, cx0, cy0)
        S, beta, cx, cy = refine_outer(pts)
        if S < best_S:
            best_S = S
            best_pts = pts.copy()
            best_beta = beta
            best_cx = cx
            best_cy = cy

    # ---------- fallback ----------
    if best_pts is None or not np.isfinite(best_S):
        best_pts = seed_pts.copy()
        best_S, best_beta, best_cx, best_cy = refine_outer(best_pts)

    cx, cy = best_cx, best_cy
    angle_deg = math.degrees(best_beta)

    # tiny safety scale about the centroid
    SCALE = 1.0 + 1e-12
    c0 = np.array([cx, cy])
    pts_scaled = c0 + (best_pts - c0) * SCALE

    # recompute exact side length
    S_final, _ = eval_exact(pts_scaled, best_beta, cx, cy)

    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)

    return inner, center, side, rotation
