import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 6.0
PI = math.pi

# ---------------------------------------------------------------------------
# Candidate point sets (axial coords).  These are *starting* rigid placements;
# per-hexagon translations are then optimized.
# ---------------------------------------------------------------------------
CANDIDATES = [
    # 3-4-3-1 tight core
    [(0, 0), (1, 0), (2, 0),
     (0, 1), (1, 1), (2, 1), (3, 1),
     (0, 2), (1, 2), (2, 2),
     (1, 3)],
    # 4-3-3
    [(0, 0), (1, 0), (2, 0), (3, 0),
     (0, 1), (1, 1), (2, 1),
     (1, 2), (2, 2), (3, 2)],
    # 4-4-3
    [(0, 0), (1, 0), (2, 0), (3, 0),
     (0, 1), (1, 1), (2, 1), (3, 1),
     (1, 2), (2, 2), (3, 2)],
    # 4-3-4-2
    [(0, 0), (1, 0), (2, 0), (3, 0),
     (1, 1), (2, 1), (3, 1),
     (0, 2), (1, 2), (2, 2), (3, 2),
     (1, 3), (2, 3)],
    # 3-4-3-1 variant
    [(0, 0), (1, 0), (2, 0),
     (0, 1), (1, 1), (2, 1), (3, 1),
     (0, 2), (1, 2), (2, 2), (3, 2)],
    # 2-4-4-1
    [(0, 0), (1, 0),
     (0, 1), (1, 1), (2, 1), (3, 1),
     (0, 2), (1, 2), (2, 2), (3, 2),
     (1, 3)],
    # 3-4-2-1-1
    [(0, 0), (1, 0), (2, 0),
     (0, 1), (1, 1), (2, 1), (3, 1),
     (1, 2), (2, 2),
     (0, 3), (3, 3)],
    # 3-3-2-2-1
    [(0, 0), (1, 0), (2, 0),
     (0, 1), (1, 1), (2, 1),
     (1, 2), (2, 2),
     (0, 3), (1, 3), (2, 3)],
    # 3-3-3-1-1
    [(0, 0), (1, 0), (2, 0),
     (0, 1), (1, 1), (2, 1),
     (0, 2), (1, 2), (2, 2),
     (0, 3), (1, 3)],
    # hex cluster
    [(0, 0),
     (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
     (1, 1), (-1, 2), (-2, 1), (2, -1)],
    # star
    [(0, 0),
     (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
     (2, 0), (-2, 0), (0, 2), (0, -2)],
    # 3x3 + 2
    [(0, 0), (1, 0), (2, 0),
     (0, 1), (1, 1), (2, 1),
     (0, 2), (1, 2), (2, 2),
     (0, 3), (1, 3)],
    # 4-2-2-2-1
    [(0, 0), (1, 0), (2, 0), (3, 0),
     (0, 1), (1, 1),
     (0, 2), (1, 2),
     (0, 3), (1, 3), (0, 4)],
    # 2-3-3-2-1
    [(0, 0), (1, 0),
     (0, 1), (1, 1), (2, 1),
     (0, 2), (1, 2), (2, 2),
     (0, 3), (1, 3), (1, 4)],
    # 5-5-1
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
     (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
     (0, 2), (1, 2)],
    # 2-5-4
    [(0, 0), (1, 0),
     (0, 1), (1, 1), (2, 1), (3, 1),
     (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
]


def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # ---------------- geometry helpers ----------------
    def axial_to_cartesian(axial):
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts, dtype=float)

    def get_normals(beta):
        return np.array([
            (math.cos(beta + PI / 6.0 + k * PI / 3.0),
             math.sin(beta + PI / 6.0 + k * PI / 3.0))
            for k in range(6)
        ])

    def wrap_beta(b):
        return ((b % (PI / 6.0)) + PI / 6.0) % (PI / 6.0)

    # exact support / D for a full configuration (center c, normals)
    def D_exact(pts, c, normals):
        centered = pts - np.asarray(c)
        projs = centered @ normals.T
        return float(np.max(np.abs(projs)))

    def S_from_D(D):
        return (D + APOTHEM) / APOTHEM

    # ---------------- smooth surrogate (log-sum-exp) ----------------
    def smooth_D(pts, c, normals, temp):
        centered = pts - np.asarray(c)
        projs = centered @ normals.T
        a = np.concatenate([projs.ravel(), (-projs).ravel()])
        m = float(np.max(a))
        s = np.clip((a - m) / temp, -60.0, 60.0)
        return temp * (math.log(float(np.sum(np.exp(s)))) + m / temp)

    def smooth_obj(pts, c, beta, temp):
        normals = get_normals(beta)
        return S_from_D(smooth_D(pts, c, normals, temp))

    # ---------------- full-config optimizer ----------------
    def optimize_full(pts0):
        """Optimize center (2D) + rotation (1D) for fixed hex positions."""
        best_S = float('inf')
        best_beta = 0.0
        best_c = np.mean(pts0, axis=0)

        def calc_S_beta(beta):
            """LP over center for fixed beta -> (S, center)."""
            normals = get_normals(beta)
            n_pts = len(pts0)
            proj = pts0 @ normals.T  # (n,6)
            A1 = np.zeros((n_pts, 6, 3)); A2 = np.zeros((n_pts, 6, 3))
            A1[:, :, 0] = -normals[0]; A1[:, :, 1] = -normals[1]; A1[:, :, 2] = -1.0
            A2[:, :, 0] = normals[0]; A2[:, :, 1] = normals[1]; A2[:, :, 2] = -1.0
            A_ub = np.concatenate([A1.reshape(-1, 3), A2.reshape(-1, 3)], axis=0)
            b_ub = np.concatenate([-proj.ravel(), proj.ravel()])
            try:
                res = linprog(np.array([0.0, 0.0, 1.0]), A_ub=A_ub, b_ub=b_ub,
                              bounds=[(None, None), (None, None), (0, None)],
                              method='highs')
                if res.status == 0:
                    return S_from_D(float(res.x[2])), (float(res.x[0]), float(res.x[1]))
            except Exception:
                pass
            # fallback
            c0 = np.mean(pts0, axis=0)

            def D_of_c(c):
                return D_exact(pts0, c, normals)
            r = minimize(D_of_c, c0, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12})
            return S_from_D(float(r.fun)), (float(r.x[0]), float(r.x[1]))

        # phase 1: coarse grid over beta
        for i in range(41):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (PI / 6.0) / 40.0
            S, c = calc_S_beta(beta)
            if S < best_S:
                best_S, best_beta, best_c = S, beta, np.asarray(c)
        # phase 2: scalar refine
        try:
            lo = max(0.0, best_beta - PI / 60.0)
            hi = min(PI / 6.0, best_beta + PI / 60.0)
            r = minimize_scalar(lambda b: calc_S_beta(b)[0], bounds=(lo, hi),
                                method='bounded', options={'maxiter': 300, 'xatol': 1e-13})
            if r.fun < best_S:
                best_S, best_beta = float(r.fun), float(r.x)
                _, best_c = calc_S_beta(best_beta)
                best_c = np.asarray(best_c)
        except Exception:
            pass
        return best_S, best_beta, best_c

    # ---------------- per-hexagon translation optimizer ----------------
    def optimize_translations(pts0, beta0, c0):
        """Optimize the 22 translation coords (11 hexes) for fixed rotation."""
        pts0 = np.asarray(pts0, dtype=float)
        n = len(pts0)

        def full_obj(theta):
            dx = theta[:n]; dy = theta[n:2 * n]
            pts = pts0.copy()
            pts[:, 0] += dx; pts[:, 1] += dy
            # keep centroid at c0 (removes translational degeneracy)
            pts[:, 0] += c0[0] - np.mean(pts[:, 0])
            pts[:, 1] += c0[1] - np.mean(pts[:, 1])
            return smooth_obj(pts, c0, beta0, 0.04)

        best_S = float('inf')
        best_theta = np.zeros(2 * n)

        # anneal temperatures
        for temp in (0.10, 0.05, 0.025):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break

            def obj_t(theta, t=temp):
                dx = theta[:n]; dy = theta[n:2 * n]
                pts = pts0.copy()
                pts[:, 0] += dx; pts[:, 1] += dy
                pts[:, 0] += c0[0] - np.mean(pts[:, 0])
                pts[:, 1] += c0[1] - np.mean(pts[:, 1])
                return smooth_obj(pts, c0, beta0, t)

            starts = [np.zeros(2 * n)]
            rng = np.random.default_rng(7)
            for _ in range(4):
                starts.append(rng.normal(0.0, 0.12, size=2 * n))
            for s in starts:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                try:
                    r = minimize(obj_t, s, method='Nelder-Mead',
                                 options={'maxiter': 400, 'xatol': 1e-12, 'fatol': 1e-12})
                    if r.fun < best_S:
                        best_S = float(r.fun)
                        best_theta = r.x
                except Exception:
                    pass

        # exact polish of the translation coords (high precision)
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            normals = get_normals(beta0)

            def exact_t(theta):
                dx = theta[:n]; dy = theta[n:2 * n]
                pts = pts0.copy()
                pts[:, 0] += dx; pts[:, 1] += dy
                pts[:, 0] += c0[0] - np.mean(pts[:, 0])
                pts[:, 1] += c0[1] - np.mean(pts[:, 1])
                return S_from_D(D_exact(pts, c0, normals))

            try:
                r = minimize(exact_t, best_theta, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-15})
                if r.fun < best_S:
                    best_S = float(r.fun)
                    best_theta = r.x
            except Exception:
                pass

        dx = best_theta[:n]; dy = best_theta[n:2 * n]
        pts = pts0.copy()
        pts[:, 0] += dx; pts[:, 1] += dy
        pts[:, 0] += c0[0] - np.mean(pts[:, 0])
        pts[:, 1] += c0[1] - np.mean(pts[:, 1])
        return best_S, pts

    # ---------------- combined (center, rotation, translations) optimizer ----------------
    def optimize_combined(pts0):
        best_S, best_beta, best_c = optimize_full(pts0)
        best_pts = pts0.copy()

        # iterate: fix rotation -> optimize translations -> re-optimize rotation
        for it in range(3):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            S2, pts2 = optimize_translations(best_pts, best_beta, best_c)
            if S2 < best_S:
                best_S, best_pts = S2, pts2
            # re-optimize rotation/center for the improved positions
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            S3, beta3, c3 = optimize_full(best_pts)
            if S3 < best_S:
                best_S, best_beta, best_c = S3, beta3, np.asarray(c3)

        # final high-precision joint polish via smooth surrogate
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            n = len(best_pts)

            def joint_obj(v):
                cx, cy = v[0], v[1]
                beta = wrap_beta(v[2])
                dx = v[3:3 + n]; dy = v[3 + n:3 + 2 * n]
                pts = best_pts.copy()
                pts[:, 0] += dx; pts[:, 1] += dy
                pts[:, 0] += cx - np.mean(pts[:, 0])
                pts[:, 1] += cy - np.mean(pts[:, 1])
                return smooth_obj(pts, (cx, cy), beta, 0.02)

            init = np.concatenate([[best_c[0], best_c[1], best_beta], np.zeros(2 * n)])
            try:
                r = minimize(joint_obj, init, method='Nelder-Mead',
                             options={'maxiter': 4000, 'xatol': 1e-13, 'fatol': 1e-14})
                if r.fun < best_S:
                    cx, cy = float(r.x[0]), float(r.x[1])
                    beta = wrap_beta(float(r.x[2]))
                    dx = r.x[3:3 + n]; dy = r.x[3 + n:3 + 2 * n]
                    pts = best_pts.copy()
                    pts[:, 0] += dx; pts[:, 1] += dy
                    pts[:, 0] += cx - np.mean(pts[:, 0])
                    pts[:, 1] += cy - np.mean(pts[:, 1])
                    best_S = S_from_D(D_exact(pts, (cx, cy), get_normals(beta)))
                    best_pts = pts
                    best_beta = beta
                    best_c = np.asarray([cx, cy])
            except Exception:
                pass
        return best_S, best_pts, best_beta, best_c

    # ---------------- main loop over candidates ----------------
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_c = None

    for axial in CANDIDATES:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts0 = axial_to_cartesian(axial)
        S, pts, beta, c = optimize_combined(pts0)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_c = c

    # ---------------- fallback ----------------
    if best_global_pts is None or not np.isfinite(best_global_S):
        best_global_pts = axial_to_cartesian(CANDIDATES[0])
        best_global_c = np.mean(best_global_pts, axis=0)
        best_global_beta = 0.0
        normals = get_normals(0.0)
        best_global_S = S_from_D(D_exact(best_global_pts, best_global_c, normals))

    # ---------------- final: recompute exact S, tiny safety scale ----------------
    cx, cy = float(best_global_c[0]), float(best_global_c[1])
    angle_deg = math.degrees(wrap_beta(best_global_beta))
    SCALE = 1.0 + 1e-10
    pts_scaled = best_global_pts.copy()
    pts_scaled[:, 0] = cx + (pts_scaled[:, 0] - cx) * SCALE
    pts_scaled[:, 1] = cy + (pts_scaled[:, 1] - cy) * SCALE

    normals = get_normals(wrap_beta(best_global_beta))
    D_scaled = D_exact(pts_scaled, (cx, cy), normals)
    S_scaled = S_from_D(D_scaled)

    inner = [[float(x), float(y), float(angle_deg)] for (x, y) in pts_scaled]
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    return inner, center, side, rotation
