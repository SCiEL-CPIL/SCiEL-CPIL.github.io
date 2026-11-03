import numpy as np
import time
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)

# 6 unit-hexagon edge normals (angles 0,60,...300)
NORM = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])

# unit hexagon vertices
UNIT = np.array([
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


def objective_flat(flat):
    C = flat.reshape(N, 2)
    side = outer_side(C)
    viol = separation_violation(C)
    return side + 100.0 * viol


def exact_nonoverlap(C, eps=1e-9):
    """Exact check: two convex polygons non-overlapping iff a separating axis exists.
    Use SAT on the 6 edge normals (both aligned)."""
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
    C = [[0.0, 0.0]]
    for k in range(6):
        a = k * np.pi / 3
        C.append([S3 * np.cos(a), S3 * np.sin(a)])
    # 4 more in a ring at radius ~2*sqrt(3)
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
    # 3-4-3-1 staggered lattice rows
    C = []
    rows = [
        (3, 0.0),
        (4, 1.5),
        (3, 3.0),
        (1, 4.5),
    ]
    for count, y in rows:
        for i in range(count):
            x = (i - (count - 1) / 2.0) * S3
            C.append([x, y])
    C = np.array(C[:N])
    return C


def init_staggered():
    C = []
    # two staggered rows of 6 and 5
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


def optimize_nm(C, deadline, maxiter=5000):
    """Local polish with Nelder-Mead."""
    flat0 = C.flatten()
    try:
        res = minimize(objective_flat, flat0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6,
                                "adaptive": True})
    except Exception:
        return C
    Cn = res.x.reshape(N, 2)
    return Cn


def construct():
    start = time.time()
    deadline = start + 140.0

    inits = [init_flower(), init_ring(), init_lattice(), init_staggered(), init_random_cluster()]

    best = None
    best_side = float('inf')

    # 1. Polish each initialization with NM
    for C in inits:
        if time.time() > deadline - 60:
            break
        Cn = optimize_nm(C, deadline, maxiter=3000)
        Sn = outer_side(Cn)
        if valid(Cn, Sn):
            if Sn < best_side:
                best_side = Sn
                best = Cn
        # Also check initial
        Si = outer_side(C)
        if valid(C, Si) and Si < best_side:
            best_side = Si
            best = C

    # 2. Polish the best found with NM again for more iterations
    if best is not None:
        if time.time() < deadline - 20:
            Cn = optimize_nm(best, deadline, maxiter=10000)
            Sn = outer_side(Cn)
            if valid(Cn, Sn) and Sn < best_side:
                best_side = Sn
                best = Cn

    # 3. If nothing valid, fallback
    if best is None:
        C = init_flower()
        Cn = optimize_nm(C, deadline, maxiter=30000)
        Sn = outer_side(Cn)
        if not valid(Cn, Sn):
            # fallback: spread out widely (valid but large)
            Cn = np.zeros((N, 2))
            for i in range(N):
                Cn[i] = [6.0 * i, 0.0]
            Sn = outer_side(Cn)
        best = Cn

    Cn = best
    inner = [[float(Cn[i, 0]), float(Cn[i, 1]), 0.0] for i in range(N)]
    return inner, (0.0, 0.0), float(best_side), 0.0
