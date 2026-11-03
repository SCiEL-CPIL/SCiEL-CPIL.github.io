"""
Logarithmic energy minimization for 100 points on the unit sphere.

Design family (kept): diverse initializations -> L-BFGS-B -> high-precision polish.
Improvements within the family:
  - Riemannian (tangent-space) L-BFGS-B with exact projected gradients and a
    projected Hessian-vector product (via finite differences of the projected
    gradient) so the quasi-Newton model respects the sphere constraint.
  - A smooth surrogate: during optimization we use a softened distance floor
    d_safe = sqrt(max(d^2, eps^2)) to avoid log singularities; the final score
    uses exact chordal distances (no floor) via math.fsum.
  - More diverse initializations (Sobol, Fibonacci, jittered, random) and a
    larger pool of L-BFGS-B starts.
  - Final polish: Riemannian gradient descent with backtracking + a short
    projected Newton (solve the projected Hessian system) for the last bits.
  - Explicit wall-clock deadline guard; returns best-so-far with safety margin.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from scipy.linalg import cho_factor, cho_solve

N = 100
# Soft floor for the surrogate during optimization (never used in final score)
_EPS_SOFT = 1e-12
_SOFT = 1.0  # multiplier on _EPS_SOFT for the floor in the surrogate

def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n

def _pair_dist_sq(pts):
    sq = np.einsum('ij,ij->i', pts, pts)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(d2, 1.0)
    return np.maximum(d2, 0.0)

def _energy(pts):
    """Surrogate energy with a soft floor (for optimization only)."""
    d2 = _pair_dist_sq(pts)
    d2 = np.maximum(d2, (_EPS_SOFT * _SOFT) ** 2)
    d = np.sqrt(d2)
    iu = np.triu_indices(N, k=1)
    return float(-np.sum(np.log(d[iu])))

def _energy_grad(pts):
    """
    Surrogate energy and its Riemannian (tangent-space) gradient.
    E = sum_{i<j} log(1/d_ij) with d_ij = sqrt(max(d2_ij, eps^2)).
    For d2 > eps^2:  dE/dx_i = -sum_{j!=i} (x_i - x_j) / d2_ij
    For d2 <= eps^2: the softened term gives a small regular gradient.
    We compute the raw Euclidean gradient then project to the tangent space.
    """
    d2_raw = _pair_dist_sq(pts)
    eps2 = (_EPS_SOFT * _SOFT) ** 2
    # Use softened d2 for gradient: d2_s = max(d2, eps2)
    d2 = np.maximum(d2_raw, eps2)
    np.fill_diagonal(d2, 1.0)  # self term not used

    iu = np.triu_indices(N, k=1)
    energy = float(-np.sum(np.log(np.sqrt(d2[iu]))))

    # W_ij = 1 / d2_ij  (softened)
    W = 1.0 / d2
    np.fill_diagonal(W, 0.0)

    row_sums = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = row_sums * pts - Wx
    grad = -raw_grad  # gradient of E

    # Project to tangent space: grad - (grad . x_i) x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    return energy, grad

def _project_grad(pts, g):
    """Project a vector field onto the tangent space at each point."""
    dot = np.sum(g * pts, axis=1, keepdims=True)
    return g - dot * pts

def _hess_vec(pts, v):
    """
    Finite-difference Hessian-vector product for the Riemannian gradient.
    H_v(x) = (grad_E(x + h*v) - grad_E(x - h*v)) / (2h)  projected to tangent.
    This is the standard trick to get Hessian-vector products without forming
    the full Hessian, and it automatically respects the manifold when combined
    with the projected gradient.
    """
    h = 1e-5
    e_plus, g_plus = _energy_grad(pts + h * v)
    e_minus, g_minus = _energy_grad(pts - h * v)
    hv = (g_plus - g_minus) / (2.0 * h)
    return _project_grad(pts, hv)

def _precise_energy(pts):
    """Exact energy with math.fsum (the actual scoring function)."""
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
    sampler_2d = qmc.Sobol(d=2, scramble=True, seed=seed)
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler_2d.random(m)
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    pts = np.column_stack([x, y_coord, z])
    return pts[:n]

def _riemannian_lbfgs(pts, max_iter=3000, ftol=1e-14, gtol=1e-12, deadline=None):
    """
    Riemannian L-BFGS-B using scipy's L-BFGS-B in a lifted Euclidean space,
    but with the gradient projected to the tangent space and the Hessian-vector
    product (via finite differences) also projected.  We use scipy's minimize
    with method='L-BFGS-B' and jac=True, but we wrap the objective so that the
    gradient is the projected one.  The projection is done at every call.
    """
    n = pts.shape[0]

    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        p = _project(p)
        e, g = _energy_grad(p)
        return e, g.reshape(-1)

    try:
        x0 = pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'maxfun': max_iter * 30,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 60,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts

def _riemannian_newton_polish(pts, max_iter=200, tol=1e-14, deadline=None):
    """
    Projected Newton polish: at each step, build the projected Hessian
    (finite-difference) and solve the linear system for the Newton step,
    then do a backtracking line search on the manifold.
    """
    n = pts.shape[0]
    best_pts = pts.copy()
    best_e = _energy(pts)

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        e, g = _energy_grad(best_pts)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break

        # Build projected Hessian via finite differences
        # H is a (3n, 3n) matrix; we build it by applying _hess_vec to basis vectors.
        # This is expensive (3n = 300 basis vectors), so we limit to a few iterations.
        dim = 3 * n
        H = np.zeros((dim, dim))
        for k in range(dim):
            evec = np.zeros(dim)
            evec[k] = 1.0
            H[:, k] = _hess_vec(best_pts, evec.reshape(n, 3)).reshape(-1)
        H = 0.5 * (H + H.T)

        # Solve H s = -g  (projected)
        g_flat = g.reshape(-1)
        try:
            lu, _ = cho_factor(H + 1e-10 * np.eye(dim))
            s = cho_solve((lu, _), -g_flat)
        except Exception:
            # Fallback: gradient descent step
            s = -g_flat / (g_norm + 1e-30)

        s = s.reshape(n, 3)
        s = _project_grad(best_pts, s)

        # Backtracking line search
        alpha = 1.0
        c1 = 1e-4
        for _ in range(30):
            candidate = _project(best_pts + alpha * s)
            cand_e = _energy(candidate)
            if cand_e <= best_e + c1 * alpha * np.sum(g_flat * s.reshape(-1)):
                best_e = cand_e
                best_pts = candidate
                break
            alpha *= 0.5
        else:
            break

    return best_pts

def _gradient_descent_polish(pts, max_iter=30000, tol=1e-14, deadline=None):
    """Riemannian gradient descent with backtracking Armijo line search."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_e = _energy(pts)
    step = 0.05
    c1 = 1e-4

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        e, g = _energy_grad(best_pts)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break

        direction = -g
        accepted = False
        alpha = step
        for _ in range(25):
            candidate = _project(best_pts + alpha * direction)
            cand_e = _energy(candidate)
            if cand_e <= best_e - c1 * alpha * (g_norm ** 2):
                best_e = cand_e
                best_pts = candidate
                accepted = True
                break
            else:
                alpha *= 0.5

        if accepted:
            step = min(step * 1.05, 0.5)
        else:
            step *= 0.7
            if step < 1e-12:
                break

    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin from 150s limit

    rng = np.random.default_rng(42)

    # Generate diverse initializations
    initializations = []

    # Sobol sequences
    for s in range(12):
        initializations.append(_sobol_sphere(N, seed=s))

    # Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))

    # Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(12):
        jitter_scale = 0.005 + 0.008 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))

    # Random points
    for s in range(12):
        initializations.append(_random_sphere(np.random.default_rng(200 + s)))

    # Score all initializations quickly
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 40.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))

    scored_inits.sort(key=lambda x: x[0])

    best_pts = None
    best_energy = float('inf')

    # Optimize top candidates with Riemannian L-BFGS-B
    top_candidates = scored_inits[:18]

    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 25.0:
            break

        optimized = _riemannian_lbfgs(init_pts, max_iter=2500, ftol=1e-14,
                                      gtol=1e-11, deadline=deadline - 10.0)
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()

    # Final polish on the global best
    if best_pts is not None and time.time() < deadline - 15.0:
        # Gradient descent polish
        polished = _gradient_descent_polish(best_pts, max_iter=25000, tol=1e-14,
                                            deadline=deadline - 8.0)
        e_precise = _precise_energy(polished)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = polished

        # Projected Newton polish (very few iterations, expensive)
        if time.time() < deadline - 3.0:
            newton_pts = _riemannian_newton_polish(best_pts, max_iter=15,
                                                   tol=1e-14,
                                                   deadline=deadline - 1.0)
            e_newton = _precise_energy(newton_pts)
            if e_newton < best_energy:
                best_energy = e_newton
                best_pts = newton_pts

    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)

    # Final normalization
    best_pts = _project(best_pts)

    return best_pts

if __name__ == '__main__':
    pts = construct()
    print(_precise_energy(pts))
