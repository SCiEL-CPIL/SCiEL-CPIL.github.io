import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
# 60-degree symmetric unit hexagon (circumradius 1)
BASE = np.array([[1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0], [0.0, 1.0],
                 [-0.5, math.sqrt(3.0) / 2.0], [-0.5, -math.sqrt(3.0) / 2.0],
                 [0.0, -1.0]])
BASE_T = BASE.T  # (6,2)
BASE_R = BASE[:, 0]
BASE_S = BASE[:, 1]

# 12 directions (unit hexagon normals + their opposites)
DIRS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 30)])
DIRS_X = DIRS[:, 0]
DIRS_Y = DIRS[:, 1]


def _outer(s, cx, cy, rot):
    """Vertices of the regular outer hexagon (side s, center (cx,cy), rotation rot rad)."""
    a = rot + np.arange(6) * (math.pi / 3.0)
    return np.column_stack([cx + s * np.cos(a), cy + s * np.sin(a)])


def _valid(inner, outer):
    """True iff every inner hexagon is strictly inside outer and pairwise disjoint."""
    n = len(inner)
    if n == 0:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            if _overlap(inner[i], inner[j]) > 1e-9:
                return False
    for i in range(n):
        if _max_viol(inner[i], outer) > 1e-9:
            return False
    return True


def _overlap(a, b):
    """Pairwise separation violation (positive = overlap)."""
    ax, ay, ar = a
    bx, by, br = b
    c, s = math.cos(ar - br), math.sin(ar - br)
    rx = c * BASE_R[1] - s * BASE_S[1]
    ry = s * BASE_R[1] + c * BASE_S[1]
    best = 1e9
    for k in range(6):
        ux = DIRS_X[k]
        uy = DIRS_Y[k]
        sep = (bx - ax) * ux + (by - ay) * uy
        sup = 1.0 + rx * ux + ry * uy
        v = 2.0 - (sep - sup)
        if v < best:
            best = v
    return best


def _max_viol(h, outer):
    """Containment violation of hexagon h relative to outer (positive = outside)."""
    x, y, r = h
    c, s = math.cos(r), math.sin(r)
    cx, cy = outer[:2]
    sside = outer[2]
    ox, oy = outer[3]
    ox = ox - cx
    oy = oy - cy
    best = 0.0
    for k in range(6):
        ux = DIRS_X[k]
        uy = DIRS_Y[k]
        sup = x * ux + y * uy + c * BASE_R[k] + s * BASE_S[k]
        supo = ox * ux + oy * uy + sside * (BASE_R[k] * math.cos(ox) + BASE_S[k] * math.sin(ox))
        v = sup - supo
        if v > best:
            best = v
    return best


def _obj(params, n, cx, cy, rot, sside, w, lam):
    P = params.reshape(n, 3)
    X = P[:, 0]
    Y = P[:, 1]
    R = P[:, 2]
    C = np.cos(R)
    S = np.sin(R)
    sup = X[:, None] * DIRS_X + Y[:, None] * DIRS_Y + C[:, None] * BASE_R + S[:, None] * BASE_S
    supo = sside * (BASE_R * math.cos(rot) + BASE_S * math.sin(rot))
    viol = sup - (cx * DIRS_X + cy * DIRS_Y + supo)
    viol = np.maximum(viol, 0.0)
    obj = w * float(np.sum(viol * viol))
    for i in range(n):
        for j in range(i + 1, n):
            dR = R[i] - R[j]
            cd = math.cos(dR)
            sd = math.sin(dR)
            rx = cd * BASE_R[1] - sd * BASE_S[1]
            ry = sd * BASE_R[1] + cd * BASE_S[1]
            dx = X[i] - X[j]
            dy = Y[i] - Y[j]
            sep = dx * DIRS_X + dy * DIRS_Y
            sup2 = 1.0 + rx * DIRS_X + ry * DIRS_Y
            v = 2.0 - (sep - sup2)
            v = np.maximum(v, 0.0)
            obj += lam * float(np.sum(v * v))
    return obj


def construct():
    t0 = time.time()
    DL = 110.0
    n = N

    def deadline():
        return (time.time() - t0) > DL

    # ---- initial guesses ----
    guesses = []
    ang0 = np.zeros(n)
    cx0, cy0 = 0.0, 0.0
    rot0 = 0.0
    s0 = 4.0
    # ring layout
    ring = np.column_stack([s0 * np.cos(2 * math.pi * np.arange(n) / n),
                            s0 * np.sin(2 * math.pi * np.arange(n) / n)]).ravel()
    guesses.append(np.concatenate([ring, ang0]))
    # hex lattice
    a3 = math.sqrt(3.0)
    hexp = []
    for k in range(-2, 3):
        for l in range(-2, 3):
            hexp.append((3.0 * k, a3 * (l + 0.5 * k)))
    hexp = hexp[:n]
    guesses.append(np.concatenate([np.array(hexp, float).ravel(), ang0]))
    # random
    rng = np.random.default_rng(7)
    guesses.append(np.concatenate([rng.uniform(-3, 3, 2 * n), rng.uniform(0, math.pi, n)]))

    w = 40000.0
    lam = 600.0
    best_s = None
    best_inner = None
    best_outer = None
    for g in guesses:
        if deadline():
            break
        try:
            r = minimize(_obj, g, args=(n, cx0, cy0, rot0, s0, w, lam),
                         method='L-BFGS-B', options={'maxiter': 200, 'gtol': 1e-5, 'maxls': 40})
            P = r.x.reshape(n, 3)
            P[:, 2] = (P[:, 2] + math.pi) % (2 * math.pi) - math.pi
            inner = [list(P[i]) for i in range(n)]
            outer = np.array([cx0, cy0, s0, rot0])
            if _valid(inner, outer):
                if best_s is None or s0 < best_s:
                    best_s, best_inner, best_outer = s0, inner, outer
        except Exception:
            pass

    if best_inner is not None:
        # ---- polish with Nelder-Mead ----
        if not deadline():
            P0 = np.array(best_inner).reshape(n, 3)
            try:
                r = minimize(_obj, P0, args=(n, best_outer[0], best_outer[1], best_outer[3],
                                             best_outer[2], w, lam),
                             method='Nelder-Mead',
                             options={'maxiter': 400, 'xatol': 1e-4, 'fatol': 1e-8, 'adaptive': True})
                P = r.x.reshape(n, 3)
                P[:, 2] = (P[:, 2] + math.pi) % (2 * math.pi) - math.pi
                inner = [list(P[i]) for i in range(n)]
                outer = np.array([best_outer[0], best_outer[1], best_outer[2], best_outer[3]])
                if _valid(inner, outer):
                    best_inner = inner
            except Exception:
                pass

        # ---- binary search on side length ----
        lo = 2.0
        hi = best_outer[2]
        if not deadline():
            for _ in range(6):
                if deadline():
                    break
                mid = 0.5 * (lo + hi)
                found = False
                for g in guesses:
                    if deadline():
                        break
                    try:
                        r = minimize(_obj, g, args=(n, best_outer[0], best_outer[1], best_outer[3],
                                                    mid, w, lam),
                                     method='L-BFGS-B', options={'maxiter': 120, 'gtol': 1e-5})
                        P = r.x.reshape(n, 3)
                        P[:, 2] = (P[:, 2] + math.pi) % (2 * math.pi) - math.pi
                        inner = [list(P[i]) for i in range(n)]
                        outer = np.array([best_outer[0], best_outer[1], mid, best_outer[3]])
                        if _valid(inner, outer):
                            hi, best_s, best_inner = mid, mid, inner
                            found = True
                            break
                    except Exception:
                        pass
                if not found:
                    lo = mid
                if hi - lo < 0.02:
                    break

    if best_inner is not None:
        return best_inner, (float(best_outer[0]), float(best_outer[1])), float(best_outer[2]), float(best_outer[3])

    # fallback: seed-like valid packing
    inner = [[3.1 * k, 0.0, 0.0] for k in range(-(n // 2), n - n // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
