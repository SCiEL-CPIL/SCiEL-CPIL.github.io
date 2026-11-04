import math
import time
import numpy as np

N = 12

# --- hexagon geometry -------------------------------------------------------
# vertices of a unit regular hexagon centered at origin, angle 0
V = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))]
              for a in range(0, 360, 60)])
# outward normals (pointing to edge midpoints)
NORM = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))]
                 for a in range(30, 390, 60)])
# support of unit hexagon in direction d = max dot(v, d)
D = np.column_stack((NORM[:, 0], NORM[:, 1]))


def support_unit(d):
    """max dot(v, d) over unit hexagon vertices (vectorized over d rows)."""
    return np.max(D @ d.T, axis=1)


def sdist(p, d, t):
    """signed distance from point p to hexagon centered c with side t, angle deg.
    positive = outside, negative = inside. Vectorized over p rows."""
    v = p - d
    c = v @ t  # cos, sin
    s = v @ t[2:]  # sin, -cos
    u = c * NORM[:, 0] + s * NORM[:, 1]
    w = c * NORM[:, 1] - s * NORM[:, 0]
    m = np.maximum.reduce([
        u, -u,
        0.8660254037844386 * w - 0.5 * u,
        -0.8660254037844386 * w - 0.5 * u,
        -0.8660254037844386 * w + 0.5 * u,
        0.8660254037844386 * w + 0.5 * u,
    ])
    return m - t[0]


def rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def vertices(center, t, theta):
    R = rot(theta)
    return center + (V * t) @ R.T


def hexagon_area(t):
    return 2.598076211353316 * t * t  # (3*sqrt3/2) t^2


def calc_L(inner, outer):
    """side length of smallest enclosing regular hexagon (fixed orientation) for
    the given inner hexagons (x, y, angle_deg)."""
    cx, cy = outer[0], outer[1]
    oang = math.radians(outer[3])
    oc, os_ = math.cos(oang), math.sin(oang)
    ON = np.array([[oc, os_], [os_, -oc]])
    pts = []
    for (x, y, a) in inner:
        pts.append(vertices((x, y), 1.0, math.radians(a)))
    P = np.vstack(pts)
    d = P - np.array([cx, cy])
    L = support_unit(d @ ON)
    return float(np.max(L))


def objective(params):
    # params: [x0,y0,ang0, ... x11,y11,ang11, cx,cy, oang]
    inner = [(params[3 * i], params[3 * i + 1], params[3 * i + 2]) for i in range(N)]
    outer = (params[3 * N], params[3 * N + 1], 0.0, params[3 * N + 2])
    # compute enclosing side
    cx, cy = outer[0], outer[1]
    oang = math.radians(outer[3])
    oc, os_ = math.cos(oang), math.sin(oang)
    ON = np.array([[oc, os_], [os_, -oc]])
    P = []
    for (x, y, a) in inner:
        P.append(vertices((x, y), 1.0, math.radians(a)))
    P = np.vstack(P)
    d = P - np.array([cx, cy])
    L = float(np.max(support_unit(d @ ON)))
    # containment: each inner vertex must be inside outer hexagon (side L)
    sd = sdist(P, np.array([cx, cy]), np.array([L, oc, os_]))
    pen = float(np.sum(np.maximum(sd, 0.0) ** 2))
    # separation: pairwise min distance between vertices
    sep_pen = 0.0
    for i in range(N):
        Pi = P[6 * i: 6 * (i + 1)]
        for j in range(i + 1, N):
            Pj = P[6 * j: 6 * (j + 1)]
            dd = Pi[:, None, :] - Pj[None, :, :]
            dist = np.sqrt((dd * dd).sum(-1)).min()
            if dist < 0.0:
                sep_pen += (0.0 - dist) ** 2
    return L + 0.5 * pen + 0.5 * sep_pen


def nelder_mead(f, x0, max_iter=4000, step=0.5):
    x = np.array(x0, dtype=float)
    n = len(x)
    xk = [x.copy() for _ in range(n + 1)]
    fk = [f(xk[0])]
    for i in range(1, n + 1):
        xk[i] = xk[0].copy()
        xk[i][i - 1] += step
        fk[i] = f(xk[i])
    order = np.argsort(fk)
    xk = [xk[order[i]] for i in range(n + 1)]
    fk = [fk[order[i]] for i in range(n + 1)]
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for _ in range(max_iter):
        centroid = sum(xk[i] for i in range(n)) / n
        xr = centroid + alpha * (centroid - xk[n])
        fr = f(xr)
        if fr < fk[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                xk[n], fk[n] = xe, fe
            else:
                xk[n], fk[n] = xr, fr
        elif fr < fk[n - 1]:
            xk[n], fk[n] = xr, fr
        else:
            xc = centroid + rho * (xk[n] - centroid)
            fc = f(xc)
            if fc < fk[n]:
                xk[n], fk[n] = xc, fc
            else:
                for i in range(1, n + 1):
                    xk[i] = xk[0] + sigma * (xk[i] - xk[0])
                    fk[i] = f(xk[i])
        order = np.argsort(fk)
        xk = [xk[order[i]] for i in range(n + 1)]
        fk = [fk[order[i]] for i in range(n + 1)]
        if abs(fk[n] - fk[0]) < 1e-12 and max(np.abs(np.array(xk[1:]) - xk[0])) < 1e-12:
            break
    return xk[0], fk[0]


def build_inner(base_angle):
    # lattice directions for a hex tiling (flat-top, 60-degree grid)
    d0 = np.array([math.cos(math.radians(base_angle)), math.sin(math.radians(base_angle))])
    d1 = np.array([math.cos(math.radians(base_angle + 60)), math.sin(math.radians(base_angle + 60))])
    d2 = np.array([math.cos(math.radians(base_angle + 120)), math.sin(math.radians(base_angle + 120))])
    # 3x4 tiling: 3 steps along d0, 4 along d1
    pos = []
    for a in range(3):
        for b in range(4):
            p = a * d0 + b * d1
            pos.append((p[0], p[1], base_angle))
    return pos


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin under 150s
    best_params = None
    best_L = float('inf')

    # initial: centered 3x4 tiling, rotation 0
    inner = build_inner(0.0)
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    params0 = []
    for p in inner:
        params0 += [p[0], p[1], p[2]]
    params0 += [cx, cy, 0.0]

    def eval_params(params):
        inner = [(params[3 * i], params[3 * i + 1], params[3 * i + 2]) for i in range(N)]
        outer = (params[3 * N], params[3 * N + 1], 0.0, params[3 * N + 2])
        return calc_L(inner, outer)

    L0 = eval_params(params0)
    best_L = L0
    best_params = list(params0)

    # try several base rotations for a good start
    for base in [0.0, 30.0, 15.0, 45.0, 60.0]:
        if time.time() > deadline:
            break
        inner = build_inner(base)
        xs = [p[0] for p in inner]
        ys = [p[1] for p in inner]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        params = []
        for p in inner:
            params += [p[0], p[1], p[2]]
        params += [cx, cy, base]
        L = eval_params(params)
        if L < best_L:
            best_L = L
            best_params = list(params)

    # refine with smooth-penalty Nelder-Mead
    if time.time() < deadline:
        xopt, _ = nelder_mead(objective, best_params, max_iter=3000, step=0.3)
        inner = [(xopt[3 * i], xopt[3 * i + 1], xopt[3 * i + 2]) for i in range(N)]
        outer = (xopt[3 * N], xopt[3 * N + 1], 0.0, xopt[3 * N + 2])
        L = calc_L(inner, outer)
        if L < best_L:
            best_L = L
            best_params = list(xopt)

    # final polish: local search over rotation + centering
    if time.time() < deadline:
        for _ in range(2):
            if time.time() > deadline:
                break
            xopt, _ = nelder_mead(objective, best_params, max_iter=1500, step=0.1)
            inner = [(xopt[3 * i], xopt[3 * i + 1], xopt[3 * i + 2]) for i in range(N)]
            outer = (xopt[3 * N], xopt[3 * N + 1], 0.0, xopt[3 * N + 2])
            L = calc_L(inner, outer)
            if L < best_L:
                best_L = L
                best_params = list(xopt)

    inner = [(best_params[3 * i], best_params[3 * i + 1], best_params[3 * i + 2]) for i in range(N)]
    outer = (best_params[3 * N], best_params[3 * N + 1], 0.0, best_params[3 * N + 2])
    L = calc_L(inner, outer)
    return inner, (outer[0], outer[1]), L, outer[3]
