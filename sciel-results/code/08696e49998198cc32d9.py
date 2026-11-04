import numpy as np
import time
import math

N = 11
S3 = np.sqrt(3.0)

# 6 unit-hexagon edge normals (angles 0,60,...,300)
NORM = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])


def outer_side(C):
    """Smallest outer hexagon (centered at origin, rot 0) side length enclosing all hexagons."""
    m = 0.0
    for k in range(6):
        n = NORM[k]
        proj = C @ n + 1.0
        m = max(m, float(proj.max()))
    return 2.0 * m / S3


def _precompute_minkowski():
    """Precompute the Minkowski-difference support table for unit hexagons.

    Two unit hexagons with relative rotation phi are disjoint iff the distance
    between their centers is >= h(phi), the support function of the Minkowski
    difference (a centrally-symmetric 12-gon). We tabulate h on a fine grid and
    interpolate linearly (it is piecewise-linear in phi)."""
    # vertices of a unit hexagon at angle 0
    hv = np.array([[np.cos(k * np.pi / 3.0), np.sin(k * np.pi / 3.0)] for k in range(6)])
    npts = 1440
    phis = np.linspace(0.0, np.pi / 3.0, npts, endpoint=False)
    h = np.empty(npts)
    for idx, phi in enumerate(phis):
        c, s = np.cos(phi), np.sin(phi)
        R = np.array([[c, -s], [s, c]])
        hv2 = hv @ R.T
        # Minkowski difference vertices = hv - hv2 (12 points)
        diff = hv[:, None, :] - hv2[None, :, :]
        diff = diff.reshape(-1, 2)
        # support function = max over outward directions of (diff . dir)
        dirs = diff / np.linalg.norm(diff, axis=1, keepdims=True)
        h[idx] = np.max(np.einsum('ij,ij->i', diff, dirs))
    return phis, h


_MPHI, _MPH = _precompute_minkowski()


def h_minkowski(phi):
    """Support h(phi) of the Minkowski difference, for relative rotation phi."""
    p = phi % (np.pi / 3.0)
    idx = int(p / (np.pi / 3.0) * len(_MPHI)) % len(_MPHI)
    return float(_MPH[idx])


def separation_violation(C, A, margin=0.0):
    """Sum of squared penetration depths over all pairs (full-orientation exact)."""
    viol = 0.0
    n = len(C)
    for i in range(n):
        for j in range(i + 1, n):
            d = C[i] - C[j]
            dist = math.hypot(d[0], d[1])
            req = h_minkowski(A[i] - A[j]) + margin
            if dist < req:
                viol += (req - dist) ** 2
    return viol


def exact_nonoverlap(C, A, eps=1e-9):
    """Exact non-overlap check for arbitrary orientations."""
    n = len(C)
    for i in range(n):
        for j in range(i + 1, n):
            d = C[i] - C[j]
            dist = math.hypot(d[0], d[1])
            if dist < h_minkowski(A[i] - A[j]) - eps:
                return False
    return True


def inside_outer(C, S, eps=1e-9):
    """All hexagons inside outer hexagon of side S centered at origin rot 0."""
    apothem = S * S3 / 2.0
    for k in range(6):
        n = NORM[k]
        proj = (C @ n) + 1.0
        if proj.max() > apothem + eps:
            return False
    return True


def valid(C, A, S):
    return exact_nonoverlap(C, A) and inside_outer(C, S)


def init_flower():
    """1+6+4 'flower' packing of 11 unit hexagons (all aligned)."""
    C = [[0.0, 0.0]]
    for k in range(6):
        a = k * np.pi / 3
        C.append([S3 * np.cos(a), S3 * np.sin(a)])
    for k in range(4):
        a = k * np.pi / 2 + np.pi / 6
        C.append([2.0 * S3 * np.cos(a), 2.0 * S3 * np.sin(a)])
    C = np.array(C[:N])
    A = np.zeros(N)
    return C, A


def init_ring():
    C = [[0.0, 0.0]]
    for k in range(10):
        a = k * 2 * np.pi / 10
        C.append([2.2 * S3 * np.cos(a), 2.2 * S3 * np.sin(a)])
    C = np.array(C[:N])
    A = np.zeros(N)
    return C, A


def init_lattice():
    C = []
    rows = [(3, 0.0), (4, 1.5), (3, 3.0), (1, 4.5)]
    for count, y in rows:
        for i in range(count):
            x = (i - (count - 1) / 2.0) * S3
            C.append([x, y])
    C = np.array(C[:N])
    A = np.zeros(N)
    return C, A


def init_staggered():
    C = []
    for i in range(6):
        C.append([(i - 2.5) * S3, 0.0])
    for i in range(5):
        C.append([(i - 2.0) * S3 + S3 / 2, 1.5])
    C = np.array(C[:N])
    A = np.zeros(N)
    return C, A


def init_random_cluster():
    rng = np.random.RandomState(0)
    C = rng.uniform(-1.5 * S3, 1.5 * S3, (N, 2))
    A = np.zeros(N)
    return C, A


def rotate(C, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return C @ R.T


def golden_section_1d(f, a, b, tol=1e-12, maxiter=200, deadline=None):
    """Minimize a unimodal function f on [a,b] via golden-section search."""
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(maxiter):
        if deadline is not None and time.time() > deadline:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = f(d)
        if abs(b - a) < tol:
            break
    x = 0.5 * (a + b)
    return x, f(x)


def _objective(C, A, tx, ty, theta):
    """Objective: outer side + strong overlap penalty (full-orientation)."""
    Cr = rotate(C, theta) + np.array([tx, ty])
    viol = separation_violation(Cr, A, margin=1e-6)
    side = outer_side(Cr)
    return side + 100.0 * viol


def optimize_full(C, A, deadline, maxiter=2500):
    """Full optimization over translation, global rotation and per-hex angles."""
    from scipy.optimize import minimize

    centroid = C.mean(axis=0)
    Cc = C - centroid

    def obj(params):
        tx, ty, theta = params[0], params[1], params[2]
        Ai = params[3:]
        return _objective(Cc, Ai, tx, ty, theta)

    best_params = np.concatenate([[0.0, 0.0, 0.0], A.copy()])
    best_val = obj(best_params)

    angle_starts = [0.0, np.pi / 12, np.pi / 6, np.pi / 4, np.pi / 3]
    for a0 in angle_starts:
        if time.time() > deadline - 20:
            break
        x0 = np.concatenate([[0.0, 0.0, a0], A.copy()])
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                           options={'maxiter': maxiter, 'xatol': 1e-9,
                                    'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    tx, ty, theta = best_params[0], best_params[1], best_params[2]
    Ai = best_params[3:]
    Cr = rotate(Cc, theta) + np.array([tx, ty])
    return Cr, Ai


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    inits = [init_flower(), init_ring(), init_lattice(),
             init_staggered(), init_random_cluster()]

    best = None
    best_side = float('inf')

    # 1. Evaluate each initialization directly.
    for C, A in inits:
        Si = outer_side(C)
        if valid(C, A, Si) and Si < best_side:
            best_side = Si
            best = (C.copy(), A.copy())

    # 2. 1D golden-section search over global rotation (phase-matching),
    #    keeping orientations fixed.
    if best is not None and time.time() < deadline - 40:
        for C, A in inits:
            if time.time() > deadline - 40:
                break
            Cc = C - C.mean(axis=0)

            def f(theta):
                return outer_side(rotate(Cc, theta))

            theta_grid = np.linspace(0.0, np.pi / 3.0, 72)
            vals = np.array([f(t) for t in theta_grid])
            kmin = int(np.argmin(vals))
            a = theta_grid[max(0, kmin - 1)]
            b = theta_grid[min(len(theta_grid) - 1, kmin + 1)]
            theta_opt, _ = golden_section_1d(f, a, b, tol=1e-13, maxiter=150,
                                             deadline=deadline)
            Cr = rotate(Cc, theta_opt) - rotate(Cc, theta_opt).mean(axis=0)
            Sn = outer_side(Cr)
            if valid(Cr, A, Sn) and Sn < best_side:
                best_side = Sn
                best = (Cr.copy(), A.copy())

    # 3. Full per-hexagon-orientation optimization (the structural change).
    if time.time() < deadline - 25:
        C0, A0 = best if best is not None else inits[0]
        Cn, An = optimize_full(C0, A0, deadline, maxiter=2500)
        Cn = Cn - Cn.mean(axis=0)
        Sn = outer_side(Cn)
        if valid(Cn, An, Sn) and Sn < best_side:
            best_side = Sn
            best = (Cn.copy(), An.copy())

    # 4. Fallback: a guaranteed-valid spread-out configuration.
    if best is None:
        C = np.zeros((N, 2))
        for i in range(N):
            C[i] = [6.0 * i, 0.0]
        A = np.zeros(N)
        best = (C, A)
        best_side = outer_side(C)

    Cn, An = best
    inner = [[float(Cn[i, 0]), float(Cn[i, 1]),
              float(np.degrees(An[i]))] for i in range(N)]
    return inner, (0.0, 0.0), float(best_side), 0.0
