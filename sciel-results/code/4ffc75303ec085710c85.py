import numpy as np
import time

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


def separation_violation(C):
    """For each pair, the max separation along the 6 directions minus 2. Negative => overlap."""
    viol = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = C[i] - C[j]
            sep = np.abs(d @ NORM.T).max()
            if sep < 2.0:
                viol += (2.0 - sep) ** 2
    return viol


def exact_nonoverlap(C, eps=1e-9):
    """SAT on the 6 edge normals (both hexagons aligned)."""
    for i in range(N):
        for j in range(i + 1, N):
            d = C[i] - C[j]
            sep = np.abs(d @ NORM.T).max()
            if sep < 2.0 - eps:
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


def valid(C, S):
    return exact_nonoverlap(C) and inside_outer(C, S)


def init_flower():
    """Known optimal 1+6+4 'flower' packing of 11 unit hexagons (exact coordinates)."""
    C = [[0.0, 0.0]]
    # 6 ring hexagons, mutually tangent, at distance sqrt(3)
    for k in range(6):
        a = k * np.pi / 3
        C.append([S3 * np.cos(a), S3 * np.sin(a)])
    # 4 outer hexagons in the 'valleys' between ring hexagons
    for k in range(4):
        a = k * np.pi / 2 + np.pi / 6
        C.append([2.0 * S3 * np.cos(a), 2.0 * S3 * np.sin(a)])
    C = np.array(C[:N])
    return C


def init_ring():
    C = [[0.0, 0.0]]
    for k in range(10):
        a = k * 2 * np.pi / 10
        C.append([2.2 * S3 * np.cos(a), 2.2 * S3 * np.sin(a)])
    C = np.array(C[:N])
    return C


def init_lattice():
    C = []
    rows = [(3, 0.0), (4, 1.5), (3, 3.0), (1, 4.5)]
    for count, y in rows:
        for i in range(count):
            x = (i - (count - 1) / 2.0) * S3
            C.append([x, y])
    C = np.array(C[:N])
    return C


def init_staggered():
    C = []
    for i in range(6):
        C.append([(i - 2.5) * S3, 0.0])
    for i in range(5):
        C.append([(i - 2.0) * S3 + S3 / 2, 1.5])
    C = np.array(C[:N])
    return C


def init_random_cluster():
    rng = np.random.RandomState(0)
    C = rng.uniform(-1.5 * S3, 1.5 * S3, (N, 2))
    return C


def rotate(C, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return C @ R.T


def golden_section_1d(f, a, b, tol=1e-13, maxiter=200, deadline=None):
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


def optimize_rigid(base_C, deadline, maxiter=2000):
    """Optimize rotation and translation of a rigid cluster (Nelder-Mead fallback)."""
    from scipy.optimize import minimize

    centroid = base_C.mean(axis=0)
    base_C_centered = base_C - centroid

    def rigid_objective(params):
        theta, tx, ty = params
        C = rotate(base_C_centered, theta) + np.array([tx, ty])
        viol = separation_violation(C)
        side = outer_side(C)
        return side + 100.0 * viol

    best_params = [0.0, 0.0, 0.0]
    best_val = rigid_objective(best_params)
    for theta0 in [0, np.pi / 12, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6]:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(rigid_objective, [theta0, 0.0, 0.0], method='Nelder-Mead',
                           options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    theta, tx, ty = best_params
    return rotate(base_C_centered, theta) + np.array([tx, ty])


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    inits = [init_flower(), init_ring(), init_lattice(), init_staggered(), init_random_cluster()]

    best = None
    best_side = float('inf')

    # 1. Evaluate each initialization directly (no heavy optimization).
    for C in inits:
        Si = outer_side(C)
        if valid(C, Si) and Si < best_side:
            best_side = Si
            best = C.copy()

    # 2. 1D golden-section search over global rotation angle (phase-matching).
    #    For each candidate cluster, sweep theta in [0, pi/3) (6-fold symmetry)
    #    to minimize the enclosing hexagon side length.
    if best is not None and time.time() < deadline - 15:
        for C in inits:
            if time.time() > deadline - 15:
                break
            centroid = C.mean(axis=0)
            Cc = C - centroid

            # Precompute rotated-projection extremes as a function of theta.
            # outer_side depends on max over k of (max_i (R(theta) Cc_i . n_k) + 1).
            # We minimize this over theta via golden section on [0, pi/3].
            def f(theta):
                Cr = rotate(Cc, theta)
                return outer_side(Cr)

            # Coarse scan to find the best sub-interval, then golden-section refine.
            theta_grid = np.linspace(0.0, np.pi / 3.0, 72)
            vals = np.array([f(t) for t in theta_grid])
            kmin = int(np.argmin(vals))
            a = theta_grid[max(0, kmin - 1)]
            b = theta_grid[min(len(theta_grid) - 1, kmin + 1)]
            theta_opt, val_opt = golden_section_1d(f, a, b, tol=1e-13, maxiter=150, deadline=deadline)

            Cr = rotate(Cc, theta_opt)
            # re-center to origin for the outer hexagon (translation doesn't change side)
            Cr = Cr - Cr.mean(axis=0)
            Sn = outer_side(Cr)
            if valid(Cr, Sn) and Sn < best_side:
                best_side = Sn
                best = Cr.copy()

    # 3. Rigid-body Nelder-Mead polish as a safety net (only if still large).
    if best is not None and best_side > 3.2 and time.time() < deadline - 20:
        Cn = optimize_rigid(best, deadline, maxiter=3000)
        Cn = Cn - Cn.mean(axis=0)
        Sn = outer_side(Cn)
        if valid(Cn, Sn) and Sn < best_side:
            best_side = Sn
            best = Cn

    # 4. Fallback: if nothing valid, use a valid spread-out configuration.
    if best is None:
        C = init_flower()
        Sn = outer_side(C)
        if valid(C, Sn):
            best = C
            best_side = Sn
        else:
            best = np.zeros((N, 2))
            for i in range(N):
                best[i] = [6.0 * i, 0.0]
            best_side = outer_side(best)

    Cn = best
    inner = [[float(Cn[i, 0]), float(Cn[i, 1]), 0.0] for i in range(N)]
    return inner, (0.0, 0.0), float(best_side), 0.0
