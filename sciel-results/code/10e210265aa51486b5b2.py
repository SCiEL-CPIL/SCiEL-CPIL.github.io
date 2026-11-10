import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 140.0


def _normalize(points):
    pts = np.asarray(points, dtype=float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _facet_alphas(points):
    """Return per-facet angular circumradius (radians). Returns [] if hull fails."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return []
    try:
        simplices = hull.simplices
    except Exception:
        return []
    if simplices.size == 0:
        return []
    tri = points[simplices]
    p0 = tri[:, 0, :]
    p1 = tri[:, 1, :]
    p2 = tri[:, 2, :]
    nrm = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(nrm, axis=1)
    mask = nn > 1e-12
    if not np.any(mask):
        return []
    d = np.abs(nrm[mask] @ p0[mask].T) / nn[mask]
    d = np.clip(d, 0.0, 1.0)
    return np.arccos(d)


def _covering_radius(points):
    """Exact covering radius in radians (max facet circumradius)."""
    alphas = _facet_alphas(points)
    if alphas.size == 0:
        return np.pi
    return float(np.max(alphas))


def _softmax(values, beta):
    if values.size == 0:
        return np.pi
    m = np.max(values)
    return m + np.log(np.sum(np.exp(values - m))) / beta


def _smooth_obj(params, beta):
    pts = _normalize(params.reshape(N, 3))
    if not np.all(np.isfinite(pts)):
        return 1e6
    alphas = _facet_alphas(pts)
    if alphas.size == 0:
        return 1e6
    return _softmax(alphas, beta)


def _smooth_obj_grad(params, beta):
    pts = _normalize(params.reshape(N, 3))
    if not np.all(np.isfinite(pts)):
        return 1e6, np.zeros_like(params)
    alphas = _facet_alphas(pts)
    if alphas.size == 0:
        return 1e6, np.zeros_like(params)
    obj = _softmax(alphas, beta)

    m = np.max(alphas)
    w = np.exp(alphas - m) / (beta * np.sum(np.exp(alphas - m)))

    try:
        hull = ConvexHull(pts)
        simplices = hull.simplices
    except Exception:
        return obj, np.zeros_like(params)

    grad = np.zeros((N, 3))
    for i, simp in enumerate(simplices):
        wi = w[i]
        if wi <= 0.0:
            continue
        a, b, c = pts[simp]
        nrm = np.cross(b - a, c - a)
        nn = np.linalg.norm(nrm)
        if nn < 1e-12:
            continue
        n = nrm / nn
        d = np.clip(np.abs(n @ a), 0.0, 1.0)
        denom = np.sqrt(max(1.0 - d * d, 1e-14))
        s = wi / denom
        ga = -s * n
        gb = -s * n
        gc = -s * n
        grad[simp[0]] += ga
        grad[simp[1]] += gb
        grad[simp[2]] += gc
    return obj, grad.flatten()


def _repulsive_relaxation(points, steps=300, alpha=0.05, time_limit=None, start_time=None):
    if start_time is None:
        start_time = time.time()
    pts = points.copy()
    for _ in range(steps):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, 1.0)
        inv_d3 = 1.0 / (d2 * np.sqrt(d2))
        forces = np.einsum('ij,ijk->ik', inv_d3, diff)
        pts = _normalize(pts + alpha * forces)
    return pts


def _fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))


def _spherical_kmeans_init(n_iter=50):
    """Spherical K-Means style initialization: start random, then iterate assignment + centroid."""
    centers = _normalize(np.random.randn(N, 3))
    for _ in range(n_iter):
        # Assign each center to nearest other center (trivial, but we do a proper k-means on sphere)
        # Actually for initialization, we just want well-spread points. Let's use a different approach:
        # Start with random, then do repulsive relaxation
        pass
    return centers


def _kmeans_sphere_init():
    """Better initialization via spherical k-means: start with random, iterate."""
    centers = _normalize(np.random.randn(N, 3))
    for _ in range(100):
        # Compute distances from each point to each center
        # For sphere, we assign to nearest center by angular distance
        # dist2[i,j] = 2 - 2 * dot(centers[i], centers[j])
        dot = centers @ centers.T
        # Assign each center to itself (trivial) - this is just for spreading
        # Better: use a set of test points or just do repulsive
        # Let's just do repulsive relaxation
        break
    return _repulsive_relaxation(centers, steps=200, alpha=0.05, time_limit=2.0)


def _local_optimize(pts, budget, start_time):
    """Local optimization: gradient on smooth surrogate + Nelder-Mead polish."""
    best = _normalize(pts)
    best_r = _covering_radius(best)

    def time_ok(b=0.5):
        return time.time() - start_time < budget - b

    # Phase A: gradient-based on smooth surrogate (coarse -> fine)
    for beta, maxiter in ((25.0, 300), (40.0, 400), (60.0, 500)):
        if not time_ok(2.0):
            break
        try:
            res = minimize(_smooth_obj, best.flatten(), method='L-BFGS-B',
                           jac=lambda x, b=beta: _smooth_obj_grad(x, b)[1],
                           options={'maxiter': maxiter, 'gtol': 1e-8,
                                    'ftol': 1e-10, 'maxfun': 2000, 'disp': False})
            if np.all(np.isfinite(res.x)):
                cand = _normalize(res.x.reshape(N, 3))
                rc = _covering_radius(cand)
                if rc < best_r:
                    best = cand
                    best_r = rc
        except Exception:
            pass

    # Phase B: Nelder-Mead polish with exact objective
    if time_ok(2.0):
        try:
            res = minimize(_covering_radius, best.flatten(), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11,
                                    'adaptive': True})
            if np.all(np.isfinite(res.x)) and res.fun < best_r:
                best = _normalize(res.x.reshape(N, 3))
                best_r = float(res.fun)
        except Exception:
            pass

    return best, best_r


def construct():
    start = time.time()
    deadline = start + _DEADLINE

    def remaining():
        return deadline - time.time()

    def time_ok(budget=0.5):
        return remaining() > budget

    # 1. Fibonacci start
    best = _fibonacci()
    best_r = _covering_radius(best)

    # 2. Repulsive relaxation from Fibonacci
    if time_ok(1.0):
        rel = _repulsive_relaxation(best, steps=1500, alpha=0.04,
                                    time_limit=6.0, start_time=start)
        rr = _covering_radius(rel)
        if rr < best_r:
            best = rel
            best_r = rr

    # 3. Local optimization
    if time_ok(10.0):
        best, best_r = _local_optimize(best, budget=25.0, start_time=start)

    # 4. Adaptive random restarts with spherical k-means style initialization
    stall = 0
    scale = 0.45
    while time_ok(15.0):
        # Random restart: perturb best + repulsive relaxation
        noise = np.random.normal(0.0, scale, size=best.shape)
        pert = _normalize(best + noise)
        rel = _repulsive_relaxation(pert, steps=250, alpha=0.05,
                                    time_limit=2.5, start_time=start)
        rr = _covering_radius(rel)
        if rr < best_r:
            best = rel
            best_r = rr
            stall = 0
            scale = max(0.08, scale * 0.85)
            if time_ok(15.0):
                best, best_r = _local_optimize(best, budget=12.0, start_time=start)
        else:
            stall += 1
            scale = min(1.0, scale * 1.15)
            if stall >= 12:
                break

    # 5. Final tight polish
    if time_ok(5.0):
        try:
            res = minimize(_covering_radius, best.flatten(), method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-11, 'fatol': 1e-13,
                                    'adaptive': True})
            if np.all(np.isfinite(res.x)) and res.fun < best_r:
                best = _normalize(res.x.reshape(N, 3))
                best_r = float(res.fun)
        except Exception:
            pass

    return best
