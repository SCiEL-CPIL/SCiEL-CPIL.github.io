"""Logarithmic energy minimization for 100 points on the unit sphere.
Combines multi-start BFGS with diverse initializations (Sobol, Fibonacci, random)."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
DEADLINE = 145.0  # safety margin before 150s kill
_START = time.time()


def _remaining():
    return DEADLINE - (time.time() - _START)


def _project(X):
    """Project points to unit sphere."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation using matrix operations."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space).
    Returns (energy, gradient) where gradient is the true gradient of the energy."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # Gradient of E = sum log(1/d) = -sum log(d)
    # dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # sum_j W_ij * (x_i - x_j) = x_i * sum_j W_ij - sum_j W_ij * x_j
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = sum_W * pts - Wx
    
    # raw_grad is sum (x_i - x_j)/d^2
    # True gradient is -raw_grad
    grad = -raw_grad
    
    # Project onto tangent space: grad - (grad . x_i) x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy."""
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
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    """Generate n random points on unit sphere."""
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence."""
    sampler_2d = qmc.Sobol(d=2, scramble=True, seed=seed)
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler_2d.random(m)
    
    # Map to sphere using spherical coordinates
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    pts = np.column_stack([x, y_coord, z])
    return pts[:n]


def _valid(X):
    """Check if configuration is valid."""
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def _optimize_bfgs(pts, max_iter=1000, ftol=1e-15, gtol=1e-12):
    """Run BFGS optimization from a starting point."""
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
        result = minimize(
            objective_and_grad,
            x0,
            method='BFGS',
            jac=True,
            options={
                'maxiter': max_iter,
                'gtol': gtol,
                'ftol': ftol,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _polish(pts, max_iter=2000, ftol=1e-16, gtol=1e-14):
    """High-precision polish pass with tighter tolerances."""
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
        result = minimize(
            objective_and_grad,
            x0,
            method='BFGS',
            jac=True,
            options={
                'maxiter': max_iter,
                'gtol': gtol,
                'ftol': ftol,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Sobol sequences (Low discrepancy) - 15 variants
    for s in range(15):
        if _remaining() < 15:
            break
        initializations.append(_sobol_sphere(N, seed=s))
    
    # 2. Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        if _remaining() < 15:
            break
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # 3. Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(8):
        if _remaining() < 15:
            break
        jitter_scale = 0.03 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # 4. Random points with different seeds
    for s in range(15):
        if _remaining() < 15:
            break
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
    
    best_pts = None
    best_energy = float('inf')
    
    # Score all initializations with fast energy
    scored_inits = []
    for init_pts in initializations:
        if _remaining() < 20:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize top candidates with BFGS
    top_candidates = scored_inits[:30]
    
    for e_quick, init_pts in top_candidates:
        if _remaining() < 10:
            break
        
        # BFGS optimization
        optimized = _optimize_bfgs(init_pts, max_iter=800, ftol=1e-14, gtol=1e-11)
        
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
    
    # Final polish on the global best with tighter tolerances
    if best_pts is not None and _remaining() > 5:
        refined = _polish(best_pts, max_iter=2000, ftol=1e-16, gtol=1e-14)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization and validation
    best_pts = _project(best_pts)
    
    # Verify no points too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Perturb slightly
                best_pts[j] += 1e-8 * rng.standard_normal(3)
                best_pts = _project(best_pts)
    
    return best_pts
