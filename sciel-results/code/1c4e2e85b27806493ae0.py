import numpy as np
import time
import math
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
# 12 separation directions (6 normals + 6 rotated by 30 deg)
N12 = np.vstack([NORM, NORM @ np.array([[0.5, -S3 / 2.0], [S3 / 2.0, 0.5]])])

# unit hexagon vertices
UNIT = np.array([
    [1.0, 0.0],
    [0.5, S3 / 2.0],
    [-0.5, S3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -S3 / 2.0],
    [0.5, -S3 / 2.0],
])


def outer_side(C, A):
    """Smallest outer hexagon (centered at origin, rot 0) side length enclosing all hexagons."""
    m = 0.0
    for k in range(6):
        n = NORM[k]
        proj = C @ n + 1.0
        m = max(m, float(proj.max()))
    return 2.0 * m / S3


def separation_violation(C, A):
    """For each pair, the max separation along the 12 directions minus 2. Negative => overlap."""
    viol = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = C[i] - C[j]
            sep = np.abs(d @ N12.T).max()
            if sep < 2.0:
                viol += (2.0 - sep) ** 2
    return viol


def objective_flat(flat):
    C = flat[0::3].reshape(N, 2)
    A = flat[1::3]
    side = outer_side(C, A)
    viol = separation_violation(C, A)
    return side + 100.0 * viol


def exact_nonoverlap(C, A, eps=1e-9):
    """Exact check: two convex polygons non-overlapping iff a separating axis exists.
    Use SAT on the 12 edge normals (union of both orientations)."""
    for i in range(N):
        for j in range(i + 1, N):
            rel = (A[i] - A[j]) * np.pi / 180.0
            c, s = np.cos(rel), np.sin(rel)
            R = np.array([[c, -s], [s, c]])
            n_j = NORM @ R
            axes = np.vstack([NORM, n_j])
            d = C[i] - C[j]
            sep = np.abs(d @ axes.T).max()
            if sep < 2.0 - eps:
                return False
    return True


def inside_outer(C, A, S, eps=1e-9):
    """All hexagons inside outer hexagon of side S centered at origin rot 0."""
    apothem = S * S3 / 2.0
    for k in range(6):
        n = NORM[k]
        proj = (C @ n) + 1.0
        if proj.max() > apothem + eps:
            return False
    return True


def valid(C, A, S):
    return exact_nonoverlap(C, A) and inside_outer(C, A, S)


def init_flower():
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


def init_rotated_flower():
    """Flower pattern with all hexagons rotated 30 degrees."""
    C, A = init_flower()
    A = np.full(N, 30.0)
    return C, A


def init_perturbed_flower(seed=0):
    """Flower pattern with small random perturbations."""
    rng = np.random.RandomState(seed)
    C, A = init_flower()
    C = C + rng.normal(0, 0.1, (N, 2))
    A = rng.uniform(-10, 10, N)
    return C, A


def optimize(C, A, deadline, maxiter=20000):
    """Multi-method local optimization: Nelder-Mead then L-BFGS-B polish."""
    flat0 = np.empty(3 * N)
    flat0[0::3] = C[:, 0]
    flat0[1::3] = A
    flat0[2::3] = C[:, 1]
    
    best_flat = flat0.copy()
    best_val = objective_flat(flat0)
    
    # Phase 1: Nelder-Mead
    if time.time() < deadline - 10:
        try:
            res = minimize(objective_flat, flat0, method="Nelder-Mead",
                           options={"maxiter": maxiter, "xatol": 1e-7, "fatol": 1e-7,
                                    "adaptive": True})
            if res.fun < best_val:
                best_val = res.fun
                best_flat = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B refinement
    if time.time() < deadline - 5:
        try:
            res2 = minimize(objective_flat, best_flat, method="L-BFGS-B",
                            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
            if res2.fun < best_val:
                best_val = res2.fun
                best_flat = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Another Nelder-Mead from refined point
    if time.time() < deadline - 3:
        try:
            res3 = minimize(objective_flat, best_flat, method="Nelder-Mead",
                            options={"maxiter": maxiter // 2, "xatol": 1e-8, "fatol": 1e-8,
                                     "adaptive": True})
            if res3.fun < best_val:
                best_val = res3.fun
                best_flat = res3.x.copy()
        except Exception:
            pass
    
    f = best_flat
    Cn = np.column_stack([f[0::3], f[2::3]])
    An = f[1::3]
    return Cn, An


def optimize_global(deadline, maxiter=2000):
    """Global optimization using differential evolution to escape local minima."""
    from scipy.optimize import differential_evolution
    bounds = [(-10, 10), (-10, 10), (-180, 180)] * N
    try:
        res = differential_evolution(objective_flat, bounds, seed=42, maxiter=maxiter,
                                     tol=1e-6, popsize=15, mutation=(0.5, 1.5),
                                     recombination=0.9, polish=True)
    except Exception:
        return None, None
    f = res.x
    Cn = np.column_stack([f[0::3], f[2::3]])
    An = f[1::3]
    return Cn, An


def coordinate_descent_polish(C, A, deadline, max_rounds=200):
    """Coordinate descent with small steps for fine polishing."""
    C = C.copy()
    A = A.copy()
    current_side = outer_side(C, A)
    
    for _round in range(max_rounds):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(N):
            # Try small perturbations in x, y, angle
            for dx in [0.01, -0.01, 0.005, -0.005]:
                C_new = C.copy()
                C_new[i, 0] += dx
                if valid(C_new, A, outer_side(C_new, A)):
                    s = outer_side(C_new, A)
                    if s < current_side:
                        C = C_new
                        current_side = s
                        improved = True
            for dy in [0.01, -0.01, 0.005, -0.005]:
                C_new = C.copy()
                C_new[i, 1] += dy
                if valid(C_new, A, outer_side(C_new, A)):
                    s = outer_side(C_new, A)
                    if s < current_side:
                        C = C_new
                        current_side = s
                        improved = True
            for da in [1.0, -1.0, 0.5, -0.5]:
                A_new = A.copy()
                A_new[i] += da
                if valid(C, A_new, outer_side(C, A_new)):
                    s = outer_side(C, A_new)
                    if s < current_side:
                        A = A_new
                        current_side = s
                        improved = True
        if not improved:
            break
    
    return C, A


def construct():
    start = time.time()
    deadline = start + 145.0

    inits = [
        init_flower(),
        init_rotated_flower(),
        init_ring(),
        init_lattice(),
        init_staggered(),
        init_perturbed_flower(0),
        init_perturbed_flower(1),
    ]

    best = None
    best_side = float('inf')

    # First pass: local optimization from multiple starts
    for C, A in inits:
        if time.time() > deadline - 30:
            break
        Cn, An = optimize(C, A, deadline, maxiter=10000)
        Sn = outer_side(Cn, An)
        if valid(Cn, An, Sn):
            if Sn < best_side:
                best_side = Sn
                best = (Cn, An, Sn)
        Si = outer_side(C, A)
        if valid(C, A, Si) and Si < best_side:
            best_side = Si
            best = (C, A, Si)

    # Second pass: global optimization to escape local minima
    if time.time() < deadline - 25:
        Cn, An = optimize_global(deadline, maxiter=800)
        if Cn is not None:
            Sn = outer_side(Cn, An)
            if valid(Cn, An, Sn) and Sn < best_side:
                best_side = Sn
                best = (Cn, An, Sn)

    # Third pass: refine best solution with more iterations
    if best is not None and time.time() < deadline - 15:
        Cn, An, Sn = best
        Cn2, An2 = optimize(Cn, An, deadline, maxiter=20000)
        Sn2 = outer_side(Cn2, An2)
        if valid(Cn2, An2, Sn2) and Sn2 < best_side:
            best_side = Sn2
            best = (Cn2, An2, Sn2)

    # Fourth pass: coordinate descent polish
    if best is not None and time.time() < deadline - 5:
        Cn, An, Sn = best
        Cn3, An3 = coordinate_descent_polish(Cn, An, deadline, max_rounds=100)
        Sn3 = outer_side(Cn3, An3)
        if valid(Cn3, An3, Sn3) and Sn3 < best_side:
            best_side = Sn3
            best = (Cn3, An3, Sn3)

    # Fallback
    if best is None:
        C, A = init_flower()
        Cn, An = optimize(C, A, deadline, maxiter=30000)
        Sn = outer_side(Cn, An)
        if not valid(Cn, An, Sn):
            Cn = np.zeros((N, 2))
            for i in range(N):
                Cn[i] = [6.0 * i, 0.0]
            An = np.zeros(N)
            Sn = outer_side(Cn, An)
        best = (Cn, An, Sn)

    Cn, An, Sn = best
    inner = [[float(Cn[i, 0]), float(Cn[i, 1]), float(An[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(Sn), 0.0
