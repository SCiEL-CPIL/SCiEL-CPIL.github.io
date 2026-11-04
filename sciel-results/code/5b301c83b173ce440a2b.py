import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0

    def hex_vertices(cx, cy, side, angle_deg):
        a = math.radians(angle_deg)
        return np.array([
            [cx + side * math.cos(a + k * math.pi / 3.0),
             cy + side * math.sin(a + k * math.pi / 3.0)] for k in range(6)
        ])

    def all_vertices(centers, angles):
        # returns (72, 2)
        pts = []
        for i in range(len(centers)):
            pts.append(hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]))
        return np.vstack(pts)

    def outer_side(centers, angles, ocx, ocy, otheta_deg):
        """Smallest regular-hexagon side (centered at ocx,ocy, rotated otheta)
        that contains all inner hexagon vertices. side = max_k (max_proj_k - min_proj_k)/sqrt(3)."""
        oth = math.radians(otheta_deg)
        pts = all_vertices(centers, angles)
        dx = pts[:, 0] - ocx
        dy = pts[:, 1] - ocy
        best = 0.0
        for k in range(6):
            na = oth + math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(na), math.sin(na)
            p = dx * nx + dy * ny
            w = p.max() - p.min()
            if w > best:
                best = w
        return best / math.sqrt(3.0)

    def outer_side_grad(centers, angles, ocx, ocy, otheta_deg):
        oth = math.radians(otheta_deg)
        pts = all_vertices(centers, angles)
        dx = pts[:, 0] - ocx
        dy = pts[:, 1] - ocy
        best = 0.0
        for k in range(6):
            na = oth + math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(na), math.sin(na)
            p = dx * nx + dy * ny
            w = p.max() - p.min()
            if w > best:
                best = w
        return best / math.sqrt(3.0)

    def is_valid(centers, angles, eps=1e-8):
        """SAT non-overlap for all pairs."""
        verts = [hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]) for i in range(12)]

        def proj(v, nx, ny):
            p = v[:, 0] * nx + v[:, 1] * ny
            return p.min(), p.max()

        def disjoint(v1, v2):
            # axes from v1
            for i in range(6):
                ex = v1[(i + 1) % 6, 0] - v1[i, 0]
                ey = v1[(i + 1) % 6, 1] - v1[i, 1]
                L = math.hypot(ex, ey)
                if L < 1e-12:
                    continue
                nx, ny = -ey / L, ex / L
                m1, M1 = proj(v1, nx, ny)
                m2, M2 = proj(v2, nx, ny)
                if M1 < m2 + eps or M2 < m1 + eps:
                    return True
            for i in range(6):
                ex = v2[(i + 1) % 6, 0] - v2[i, 0]
                ey = v2[(i + 1) % 6, 1] - v2[i, 1]
                L = math.hypot(ex, ey)
                if L < 1e-12:
                    continue
                nx, ny = -ey / L, ex / L
                m1, M1 = proj(v1, nx, ny)
                m2, M2 = proj(v2, nx, ny)
                if M1 < m2 + eps or M2 < m1 + eps:
                    return True
            return False

        for i in range(12):
            for j in range(i + 1, 12):
                if not disjoint(verts[i], verts[j]):
                    return False
        return True

    def polish_outer(centers, angles, ocx, ocy, otheta):
        """High-precision local minimization of outer side over (ocx, ocy, otheta)."""
        best_side = outer_side(centers, angles, ocx, ocy, otheta)
        bx, by, bt = ocx, ocy, otheta

        # Nelder-Mead
        try:
            def obj(x):
                return outer_side(centers, angles, x[0], x[1], x[2])
            res = minimize(obj, [bx, by, bt], method='Nelder-Mead',
                           options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 2000,
                                    'adaptive': True})
            if res.fun < best_side:
                bx, by, bt = res.x[0], res.x[1], res.x[2]
                best_side = res.fun
        except Exception:
            pass

        # Multi-scale golden-section refinement on each coordinate (with re-opt of the other two)
        for scale in [0.05, 0.01, 0.002, 0.0004, 0.00008, 0.000016]:
            if time.time() > deadline - 5:
                break
            for _ in range(40):
                improved = False
                # x
                for dx in [-scale, scale]:
                    s = outer_side(centers, angles, bx + dx, by, bt)
                    if s < best_side - 1e-13:
                        bx += dx; best_side = s; improved = True
                # y
                for dy in [-scale, scale]:
                    s = outer_side(centers, angles, bx, by + dy, bt)
                    if s < best_side - 1e-13:
                        by += dy; best_side = s; improved = True
                # theta (deg)
                for da in [-scale * 20, scale * 20]:
                    s = outer_side(centers, angles, bx, by, bt + da)
                    if s < best_side - 1e-13:
                        bt += da; best_side = s; improved = True
                if not improved:
                    break
        return bx, by, bt, best_side

    def optimize_inner(centers, angles, ocx, ocy, otheta):
        """SLSQP on inner positions/angles with soft non-overlap + containment,
        objective = outer_side."""
        n = 12
        x0 = []
        for i in range(n):
            x0.extend([centers[i][0], centers[i][1], angles[i]])

        def obj(x):
            if time.time() > deadline - 10:
                return 1e8
            c = [[x[i * 3], x[i * 3 + 1]] for i in range(n)]
            a = [x[i * 3 + 2] for i in range(n)]
            s = outer_side(c, a, ocx, ocy, otheta)
            # penalty: overlap via min pair distance (soft)
            pen = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i * 3] - x[j * 3]
                    dy = x[i * 3 + 1] - x[j * 3 + 1]
                    d = math.hypot(dx, dy)
                    if d < 2.0:
                        pen += (2.0 - d) ** 2
            return s + 1e3 * pen

        bounds = []
        for i in range(n):
            bounds.extend([(-4, 4), (-4, 4), (0, 60)])
        try:
            res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-12})
            if res.success or res.fun < obj(x0):
                newc = [[res.x[i * 3], res.x[i * 3 + 1]] for i in range(n)]
                newa = [res.x[i * 3 + 2] for i in range(n)]
                if is_valid(newc, newa):
                    return newc, newa
        except Exception:
            pass
        return centers, angles

    # ---- Initial configurations ----
    configs = []

    # 1) Hexagonal lattice: center + ring1 + 5 from ring2 (tight, valid)
    d = math.sqrt(3) + 0.02
    u = np.array([d, 0.0])
    v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
    basis = [u, v, v - u, -u, -v, u - v]
    ring1 = [u, v, v - u, -u, -v, u - v]
    centers = [np.array([0.0, 0.0])]
    centers.extend([r.copy() for r in ring1])
    ring2 = []
    for r in ring1:
        for b in basis:
            p = r + b
            if np.linalg.norm(p) < 1e-9:
                continue
            if any(np.allclose(p, r2) for r2 in ring1):
                continue
            if any(np.allclose(p, r2) for r2 in ring2):
                continue
            ring2.append(p)
    ring2.sort(key=lambda p: np.linalg.norm(p))
    centers.extend([r.copy() for r in ring2[:5]])
    angles = [30.0] * 12
    configs.append(([[c[0], c[1]] for c in centers], angles))

    # 2) Compact "flower" with slight radial compression
    for comp in [0.97, 0.95, 0.93]:
        d2 = d * comp
        u2 = np.array([d2, 0.0])
        v2 = np.array([d2 / 2.0, d2 * math.sqrt(3) / 2.0])
        b2 = [u2, v2, v2 - u2, -u2, -v2, u2 - v2]
        r1 = [u2, v2, v2 - u2, -u2, -v2, u2 - v2]
        cs = [np.array([0.0, 0.0])]
        cs.extend([r.copy() for r in r1])
        r2 = []
        for r in r1:
            for b in b2:
                p = r + b
                if np.linalg.norm(p) < 1e-9:
                    continue
                if any(np.allclose(p, r2j) for r2j in r1):
                    continue
                if any(np.allclose(p, r2j) for r2j in r2):
                    continue
                r2.append(p)
        r2.sort(key=lambda p: np.linalg.norm(p))
        cs.extend([r.copy() for r in r2[:5]])
        if is_valid([[c[0], c[1]] for c in cs], [30.0] * 12):
            configs.append(([[c[0], c[1]] for c in cs], [30.0] * 12))

    # 3) Random starts
    random.seed(7)
    for _ in range(4):
        cs = [[random.uniform(-2.5, 2.5), random.uniform(-2.5, 2.5)] for _ in range(12)]
        an = [random.uniform(0, 60) for _ in range(12)]
        if is_valid(cs, an):
            configs.append((cs, an))

    best = None
    best_val = float('inf')

    for idx, (centers, angles) in enumerate(configs):
        if time.time() > deadline - 25:
            break

        # Optimize outer hexagon for this inner config
        ocx, ocy, otheta = 0.0, 0.0, 0.0
        ocx, ocy, otheta, side = polish_outer(centers, angles, ocx, ocy, otheta)

        # Then optimize inner (positions/angles) with outer fixed
        centers2, angles2 = optimize_inner(centers, angles, ocx, ocy, otheta)

        # Re-polish outer
        ocx, ocy, otheta, side = polish_outer(centers2, angles2, ocx, ocy, otheta)

        if is_valid(centers2, angles2) and side < best_val:
            best_val = side
            best = (centers2, angles2, ocx, ocy, otheta, side)

        # Also try a second inner pass with a perturbed outer start
        if time.time() < deadline - 30:
            for da in [5.0, -5.0, 10.0, 30.0]:
                if time.time() > deadline - 28:
                    break
                oc2x, oc2y, oc2t, s2 = polish_outer(centers2, angles2, 0.0, 0.0, otheta + da)
                c3, a3 = optimize_inner(centers2, angles2, oc2x, oc2y, oc2t)
                oc2x, oc2y, oc2t, s2 = polish_outer(c3, a3, oc2x, oc2y, oc2t)
                if is_valid(c3, a3) and s2 < best_val:
                    best_val = s2
                    best = (c3, a3, oc2x, oc2y, oc2t, s2)

    if best is None:
        centers, angles = configs[0]
        ocx, ocy, otheta, side = polish_outer(centers, angles, 0.0, 0.0, 0.0)
        best = (centers, angles, ocx, ocy, otheta, side)
        best_val = side

    centers, angles, ocx, ocy, otheta, side = best
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (ocx, ocy)
    outer_side = side
    outer_angle = otheta
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
