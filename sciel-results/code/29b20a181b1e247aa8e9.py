"""Logarithmic energy minimization for N points on the unit sphere.
Uses multi-start L-BFGS-B with Riemannian gradient, diverse initializations,
and adaptive multi-stage refinement."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_DEADLINE = 142.0  # safety margin from 150s limit
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and projected gradient on sphere.
    Uses efficient single-matrix-product formulation."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # W_ij = 1/d_ij^2 for i != j
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # Gradient: grad_i = -sum_j W_ij (x_i - x_j)
    # = -x_i * sum_j W_ij + sum_j W_ij x_j
    # = -(sum_W_i) * x_i + (W @ pts)_i
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = Wx - sum_W * pts  # This is -sum_j W_ij(x_i - x_j)
    
    # True gradient of energy = raw_grad (since dE/dx_i = sum_j (x_i-x_j)/d^2 = raw_grad)
    # Wait: E = -sum log(d), dE/dx_i = -sum_j (1/d_ij) * (x_i-x_j)/d_ij = -sum_j (x_i-x_j)/d_ij^2
    # So gradient = -raw_grad where raw_grad = sum_j W_ij(x_i - x_j)
    # Let me recompute: raw_grad_i = Wx_i - sum_W_i * x_i = sum_j W_ij x_j - (sum_j W_ij) x_i
    # = sum_j W_ij (x_j - x_i) = -sum_j W_ij (x_i - x_j)
    # So -raw_grad = sum_j W_ij (x_i - x_j) = gradient of E
    grad = -raw_grad
    
    # Project onto tangent space
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
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence."""
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


def _optimize(pts, max_iter=2000, ftol=1e-15, gtol=1e-12, maxls=50, m=50):
    """Run L-BFGS-B optimization from a starting point."""
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
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'gtol': gtol,
                'ftol': ftol,
                'maxls': maxls,
                'm': m,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _valid(X):
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


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Sobol sequences (Low discrepancy) - 20 seeds
    for s in range(20):
        initializations.append(_sobol_sphere(N, seed=s))
        
    # 2. Fibonacci spirals with various offsets
    for offset in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
        
    # 3. Jittered Fibonacci spirals - 12 variants with varying scales
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(12):
        jitter_scale = 0.02 + 0.015 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
        
    # 4. Random points with different seeds - 20 seeds
    for s in range(20):
        initializations.append(_random_sphere(np.random.default_rng(200 + s)))
    
    # 5. Partially optimized Sobol (quick pre-optimize) - 5 seeds
    for s in range(5):
        init = _sobol_sphere(N, seed=1000 + s)
        # Quick pre-optimization to get closer to local minimum
        pre_opt = _optimize(init, max_iter=200, ftol=1e-12, gtol=1e-10)
        initializations.append(pre_opt)
    
    best_pts = None
    best_energy = float('inf')
    
    # Score all initializations
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 40.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
        
    scored_inits.sort(key=lambda x: x[0])
    
    # Adaptive optimization: try more candidates early, fewer later
    # Stage 1: Optimize top candidates with moderate settings
    top_candidates = scored_inits[:min(60, len(scored_inits))]
    
    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 25.0:
            break
        
        optimized = _optimize(init_pts, max_iter=1200, ftol=1e-14, gtol=1e-11)
        
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
            
    # Stage 2: Polish the best result with tighter tolerances
    if best_pts is not None and time.time() < deadline - 10.0:
        refined = _optimize(best_pts, max_iter=3000, ftol=1e-16, gtol=1e-14, m=100)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
    
    # Stage 3: Ultra-precise final polish
    if best_pts is not None and time.time() < deadline - 5.0:
        refined2 = _optimize(best_pts, max_iter=5000, ftol=1e-18, gtol=1e-16, m=150, maxls=100)
        e_precise = _precise_energy(refined2)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined2
    
    # Stage 4: If time remains, try second-best candidate with full polish
    if best_pts is not None and time.time() < deadline - 8.0 and len(top_candidates) > 1:
        # Try the second best initial
        second_best_init = top_candidates[1][1]
        if not np.array_equal(second_best_init, best_pts):
            refined3 = _optimize(second_best_init, max_iter=2000, ftol=1e-16, gtol=1e-14, m=100)
            e_precise = _precise_energy(refined3)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = refined3
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization
    best_pts = _project(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
