"""Logarithmic energy minimization for N points on the unit sphere.
Combines low-discrepancy (Sobol) + diverse initializations with a smooth
Riemannian gradient-flow polish on the sphere (tangent-space projected
gradient + adaptive line search), giving a continuous smooth optimization
that avoids combinatorial point re-assignment."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_HARD_LIMIT = 150.0
_SAFETY = 20.0
_START = time.time()


def _remaining():
    return _HARD_LIMIT - _SAFETY - (time.time() - _START)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _pairwise_dist_sq(pts):
    sq = np.einsum('ij,ij->i', pts, pts)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(d2, 1.0)
    return np.maximum(d2, 1e-30)


def _energy(pts):
    """Fast energy: sum_{i<j} log(1/d_ij)."""
    d2 = _pairwise_dist_sq(pts)
    iu = np.triu_indices(pts.shape[0], k=1)
    return float(-np.sum(np.log(np.sqrt(d2[iu]))))


def _energy_grad(pts):
    """Energy and tangent-projected gradient.
    E = -sum_{i<j} log d_ij.  dE/dx_i = sum_{j!=i} (x_i - x_j)/d_ij^2.
    Returns (energy, grad) with grad projected onto the tangent space at each x_i."""
    d2 = _pairwise_dist_sq(pts)
    iu = np.triu_indices(pts.shape[0], k=1)
    energy = float(-np.sum(np.log(np.sqrt(d2[iu]))))

    W = 1.0 / d2
    np.fill_diagonal(W, 0.0)
    row_sums = W.sum(axis=1, keepdims=True)
    grad = row_sums * pts - W @ pts          # sum_{j!=i} (x_i - x_j)/d_ij^2
    # Project onto tangent space: remove radial component
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    return energy, grad


def _precise_energy(pts):
    """Energy using math.fsum for accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_sphere(n, seed=0):
    """Low-discrepancy points on the sphere from a 2D Sobol sequence."""
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler.random(m)
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    pts = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    return pts[:n]


def _optimize_lbfgs(pts, max_iter=1500, ftol=1e-15, gtol=1e-12):
    """L-BFGS-B on flattened coords with re-projection inside the objective."""
    n = pts.shape[0]

    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        e, g = _energy_grad(p)
        return e, g.reshape(-1)

    try:
        x0 = pts.reshape(-1)
        res = minimize(
            objective_and_grad, x0,
            method='L-BFGS-B', jac=True,
            options={'maxiter': max_iter, 'maxfun': max_iter * 10,
                     'ftol': ftol, 'gtol': gtol, 'maxls': 50},
        )
        return _project(res.x.reshape(n, 3))
    except Exception:
        return pts


def _riemannian_flow(pts, max_iter=4000, tol=1e-14, max_step=0.5):
    """Smooth continuous optimization on the sphere: tangent gradient descent
    with adaptive step + backtracking line search. This moves points along the
    steepest-descent direction of the smooth log-energy field."""
    best = pts.copy()
    best_e = _energy(best)
    step = 0.05
    for _ in range(max_iter):
        e, g = _energy_grad(best)
        gn = np.linalg.norm(g)
        if gn < tol:
            break
        g = g / gn                      # normalize direction
        # adaptive step with backtracking (Armijo)
        s = step
        cand = _project(best - s * g)
        cand_e = _energy(cand)
        improved = False
        for _ls in range(20):
            if cand_e < best_e - 1e-15:
                best, best_e, improved = cand, cand_e, True
                break
            s *= 0.5
            if s < 1e-12:
                break
            cand = _project(best - s * g)
            cand_e = _energy(cand)
        if improved:
            step = min(step * 1.2, max_step)
        else:
            step *= 0.5
            if step < 1e-12:
                break
    return best


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = _pairwise_dist_sq(X)
    np.fill_diagonal(d2, np.inf)
    return np.min(d2) >= (1e-9) ** 2


def construct():
    start = time.time()
    deadline = start + 125.0           # 25s safety margin before 150s hard kill
    rng = np.random.default_rng(42)

    inits = []

    # 1) Sobol low-discrepancy starts (strongest measured improvement source)
    for s in range(20):
        inits.append(_sobol_sphere(N, seed=s))

    # 2) Fibonacci spirals + jittered + rotated variants
    base = _fibonacci_spiral(N, offset=0.5)
    for off in [0.1, 0.3, 0.5, 0.7, 0.9]:
        inits.append(_fibonacci_spiral(N, offset=off))
    for s in range(12):
        js = 0.02 + 0.012 * s
        inits.append(_project(base + js * rng.standard_normal((N, 3))))
    for ang in [0.3, 0.7, 1.1, 1.5, 2.0]:
        rot = np.array([
            [np.cos(ang), -np.sin(ang), 0],
            [np.sin(ang), np.cos(ang), 0],
            [0, 0, 1],
        ])
        inits.append(_project(base @ rot.T))

    # 3) Random starts
    for s in range(15):
        inits.append(_random_sphere(np.random.default_rng(200 + s)))

    # ---- Pass 1: quick score, keep top candidates ----
    scored = []
    for ip in inits:
        if time.time() > deadline - 25.0:
            break
        scored.append((_energy(ip), ip))
    scored.sort(key=lambda x: x[0])

    best_pts, best_e = None, float('inf')
    top = scored[:25]
    for _, ip in top:
        if time.time() > deadline - 15.0:
            break
        opt = _optimize_lbfgs(ip, max_iter=1200, ftol=1e-14, gtol=1e-10)
        pe = _precise_energy(opt)
        if pe < best_e:
            best_e, best_pts = pe, opt

    # ---- Pass 2: smooth Riemannian flow polish on the global best ----
    if best_pts is not None and time.time() < deadline - 8.0:
        flowed = _riemannian_flow(best_pts, max_iter=4000, tol=1e-14)
        pe = _precise_energy(flowed)
        if pe < best_e:
            best_e, best_pts = pe, flowed

    # ---- Pass 3: final tight L-BFGS-B polish ----
    if best_pts is not None and time.time() < deadline - 3.0:
        refined = _optimize_lbfgs(best_pts, max_iter=3000, ftol=1e-16, gtol=1e-13)
        pe = _precise_energy(refined)
        if pe < best_e:
            best_e, best_pts = pe, refined

    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)

    best_pts = _project(best_pts)
    if not _valid(best_pts) or not np.isfinite(_precise_energy(best_pts)):
        best_pts = _project(_fibonacci_spiral(N, offset=0.5))
    return best_pts
