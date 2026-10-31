import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)


def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(points):
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _soft_min_value(values, beta):
    """Soft minimum using log-sum-exp trick."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    vmin = np.min(values)
    return vmin - np.log(np.sum(np.exp(-(values - vmin) * beta))) / beta


def _soft_score(points, beta):
    """Soft version of the score for gradient-based optimization."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    soft_min = _soft_min_value(areas, beta)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(points, beta):
    s = _soft_score(points, beta)
    if s <= 0:
        return 1e10
    return -s


# ---------- Base topologies ----------

def _base_regular():
    """Regular 13-gon on unit circle."""
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _base_grid_4x4():
    """4x4 grid centered, minus one corner (13 points)."""
    xs = np.array([-1.5, -0.5, 0.5, 1.5])
    ys = np.array([-1.5, -0.5, 0.5, 1.5])
    grid = []
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 0:
                continue  # skip one corner to get 13
            grid.append([xs[i], ys[j]])
    return np.array(grid, dtype=np.float64)


def _base_grid_3x3_plus_4():
    """3x3 grid (9 points) plus 4 points on a larger ring."""
    xs = np.array([-1.0, 0.0, 1.0])
    ys = np.array([-1.0, 0.0, 1.0])
    grid = []
    for i in range(3):
        for j in range(3):
            grid.append([xs[i], ys[j]])
    # 4 points on a ring of radius 2.0
    ring_ang = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    for a in ring_ang:
        grid.append([2.0 * np.cos(a), 2.0 * np.sin(a)])
    return np.array(grid, dtype=np.float64)


def _base_two_ring(n_out=7, r_in=0.55, offset=np.pi/7):
    """Two concentric rings."""
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(offset, 2 * np.pi + offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_in * np.cos(a_in), r_in * np.sin(a_in)])
    return np.vstack([outer, inner])


def _base_three_ring(n1=5, n2=4, n3=4, r2=0.6, r3=0.3):
    """Three concentric rings."""
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(np.pi/n1, 2*np.pi + np.pi/n1, n2, endpoint=False)
    a3 = np.linspace(np.pi/n3, 2*np.pi + np.pi/n3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _base_quasi_random(seed=42):
    """Quasi-random points from a low-discrepancy sequence."""
    rng = np.random.RandomState(seed)
    # Use golden-ratio-based quasi-random
    golden = (1 + np.sqrt(5)) / 2
    angles = np.array([(i * (1.0/golden)) % 1.0 for i in range(N)]) * 2 * np.pi
    # Add slight radial variation
    radii = 0.8 + 0.4 * np.abs(np.sin(angles * 2 + 1.0))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _base_perturbed_regular(seed=0, sigma=0.05):
    """Regular 13-gon with small random perturbations."""
    base = _base_regular()
    rng = np.random.RandomState(seed)
    return base + sigma * rng.randn(N, 2)


# ---------- Affine transform helpers ----------

def _apply_affine(base, A_flat):
    """Apply affine transform (2x2 matrix) to base points.
    A_flat: 4-element array [a, b, c, d] representing [[a,b],[c,d]]
    """
    A = A_flat.reshape(2, 2)
    return base @ A.T


def _neg_soft_affine(base, beta):
    """Objective for affine optimization."""
    def obj(A_flat):
        pts = _apply_affine(base, A_flat)
        return _neg_soft_score(pts, beta)
    return obj


def _neg_affine_true(base):
    """Objective for affine optimization using true score."""
    def obj(A_flat):
        pts = _apply_affine(base, A_flat)
        return _neg_score(pts)
    return obj


# ---------- Annealed soft-min optimization over affine parameters ----------

def _annealed_soft_affine(base, deadline, initial_beta=30.0, final_beta=500.0,
                          n_anneal_steps=8, max_iters_per_step=300):
    """
    Anneal soft-min temperature while optimizing the 4 affine parameters.
    """
    A0 = np.array([1.0, 0.0, 0.0, 1.0])  # identity
    best_A = A0.copy()
    best_pts = _apply_affine(base, A0)
    best_s = _score(best_pts)

    betas = np.geomspace(initial_beta, final_beta, n_anneal_steps)

    for beta in betas:
        if time.time() >= deadline - 5:
            break

        obj = _neg_soft_affine(base, beta)

        # L-BFGS-B
        try:
            res = minimize(obj, best_A, method='L-BFGS-B',
                          options={'maxiter': max_iters_per_step, 'gtol': 1e-10, 'ftol': 1e-12})
            cand_pts = _apply_affine(base, res.x)
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best_A = res.x.copy()
                best_pts = cand_pts
        except Exception:
            pass

        # Powell for robustness
        if time.time() >= deadline - 5:
            break
        try:
            res = minimize(obj, best_A, method='Powell',
                          options={'maxiter': max_iters_per_step // 2, 'xatol': 1e-10, 'fatol': 1e-10})
            cand_pts = _apply_affine(base, res.x)
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best_A = res.x.copy()
                best_pts = cand_pts
        except Exception:
            pass

    return best_pts, best_s


# ---------- Full-coordinate polish ----------

def _optimize_full(points, deadline, max_iters_powell=400):
    """Polish a point configuration using Powell then Nelder-Mead on the true objective."""
    best = points.copy()
    best_s = _score(best)

    # Powell on the true (non-smooth) objective
    if time.time() < deadline - 8:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Powell',
                options={'maxiter': max_iters_powell, 'xatol': 1e-9, 'fatol': 1e-9},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    # Nelder-Mead fine polish
    if time.time() < deadline - 4:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    return best, best_s


# ---------- Full-coordinate annealed soft-min (for final refinement) ----------

def _annealed_soft_full(points, deadline, initial_beta=50.0, final_beta=800.0,
                        n_anneal_steps=6, max_iters_per_step=200):
    """Annealed soft-min over full 26-dimensional coordinates."""
    best = points.copy()
    best_s = _score(best)

    betas = np.geomspace(initial_beta, final_beta, n_anneal_steps)

    for beta in betas:
        if time.time() >= deadline - 5:
            break

        def obj(x):
            pts = x.reshape(N, 2)
            return _neg_soft_score(pts, beta)

        try:
            res = minimize(obj, best.flatten(), method='L-BFGS-B',
                          options={'maxiter': max_iters_per_step, 'gtol': 1e-9, 'ftol': 1e-12})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand
        except Exception:
            pass

        if time.time() >= deadline - 5:
            break
        try:
            res = minimize(obj, best.flatten(), method='Powell',
                          options={'maxiter': max_iters_per_step // 2, 'xatol': 1e-9, 'fatol': 1e-9})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand
        except Exception:
            pass

    return best, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s kill

    np.random.seed(42)

    best_points = None
    best_score = 0.0

    def _update(pts):
        nonlocal best_points, best_score
        if pts is None or len(pts) != N:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # ---------- Generate diverse base topologies ----------
    bases = []

    # Regular polygon
    bases.append(_base_regular())

    # Grid topologies
    bases.append(_base_grid_4x4())
    bases.append(_base_grid_3x3_plus_4())

    # Ring configurations
    for n_out in [6, 7, 8]:
        for r_in in [0.45, 0.55, 0.65]:
            for a_off in [0.0, np.pi / n_out, np.pi / (2.0 * n_out)]:
                bases.append(_base_two_ring(n_out, r_in, a_off))

    # Three-ring
    bases.append(_base_three_ring(5, 4, 4, 0.6, 0.3))
    bases.append(_base_three_ring(6, 4, 3, 0.55, 0.25))

    # Quasi-random
    for seed in [42, 123, 999]:
        bases.append(_base_quasi_random(seed))

    # Perturbed regular
    for seed in [0, 1, 2]:
        bases.append(_base_perturbed_regular(seed, sigma=0.04))

    # ---------- Evaluate all base seeds directly ----------
    for base in bases:
        _update(base)

    if best_points is None:
        best_points = _base_regular()
        best_score = _score(best_points)

    # ---------- Affine-optimized annealing from multiple base topologies ----------
    # Pick the top diverse bases for affine optimization
    # We'll try affine optimization from several bases
    affine_bases = [
        _base_regular(),
        _base_grid_4x4(),
        _base_grid_3x3_plus_4(),
        _base_two_ring(7, 0.55, np.pi/7),
        _base_three_ring(5, 4, 4, 0.6, 0.3),
        _base_quasi_random(42),
        _base_perturbed_regular(0, 0.04),
    ]

    for base in affine_bases:
        if time.time() >= deadline - 25:
            break
        try:
            pts, s = _annealed_soft_affine(base, deadline,
                                           initial_beta=20.0, final_beta=400.0,
                                           n_anneal_steps=6, max_iters_per_step=200)
            _update(pts)
        except Exception:
            pass

    # ---------- Full-coordinate annealed soft-min from best so far ----------
    if best_points is not None and time.time() < deadline - 15:
        try:
            pts, s = _annealed_soft_full(best_points, deadline,
                                         initial_beta=40.0, final_beta=600.0,
                                         n_anneal_steps=5, max_iters_per_step=150)
            _update(pts)
        except Exception:
            pass

    # ---------- Full-coordinate polish ----------
    if best_points is not None and time.time() < deadline - 12:
        try:
            pts, s = _optimize_full(best_points, deadline, max_iters_powell=400)
            _update(pts)
        except Exception:
            pass

    # ---------- Final ultra-fine Nelder-Mead ----------
    if best_points is not None and time.time() < deadline - 4:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass

    if best_points is None:
        best_points = _base_regular()

    return best_points
