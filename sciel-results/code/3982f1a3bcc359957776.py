"""
Logarithmic energy minimization for N points on the unit sphere.
Combines vectorized gradient descent (current) with multi-start BFGS polish (inspiration).
Uses Sobol and Fibonacci initializations for diverse starts.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100

def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for precision."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dists[mask]
    if np.any(d < 1e-9):
        return float('inf')
    terms = -np.log(d)
    return math.fsum(terms.tolist())

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space)."""
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
    return _log_energy(pts)

def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _normalize(v)

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

def _gradient_descent(pts, deadline, max_iters=10000):
    """Vectorized gradient descent with step decay."""
    step_size = 0.1
    current_pts = pts.copy()
    best_energy = _log_energy(current_pts)
    best_pts = current_pts.copy()
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute pairwise differences and distances
        diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0)
        
        inv_d2 = 1.0 / (dists * dists)
        inv_d2[np.diag_indices(N)] = 0.0
        
        # Gradient at i is sum_j diff[i,j] * inv_d2[i,j]
        grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        # Project gradient to tangent space
        dots = np.sum(grad * current_pts, axis=1)
        proj_grad = grad - dots[:, np.newaxis] * current_pts
        
        # Update
        current_pts = current_pts - step_size * proj_grad
        current_pts = _normalize(current_pts)
        
        # Compute energy
        energy = _log_energy(current_pts)
        if energy < best_energy:
            best_energy = energy
            best_pts = current_pts.copy()
            
        # Decay step size
        step_size *= 0.995
        
        if step_size < 1e-8:
            break
            
    return best_pts

def _bfgs_optimize(pts, deadline, max_iter=500):
    """Run BFGS optimization from a starting point."""
    n = pts.shape[0]
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        p = _normalize(p)
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
                'gtol': 1e-12,
                'ftol': 1e-12,
            }
        )
        new_pts = _normalize(result.x.reshape(n, 3))
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
    deadline = start_time + 110.0  # 10 second safety margin before 120s kill
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Sobol sequences (Low discrepancy)
    for s in range(10):
        initializations.append(_sobol_sphere(N, seed=s))
        
    # 2. Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
        
    # 3. Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(5):
        jitter_scale = 0.05 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_normalize(jittered))
        
    # 4. Random points with different seeds
    for s in range(10):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Gradient descent on top initializations
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        e_quick = _log_energy(init_pts)
        scored_inits.append((e_quick, init_pts))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize top candidates with gradient descent
    top_candidates = scored_inits[:10]
    
    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 20.0:
            break
        
        # Gradient descent first
        gd_result = _gradient_descent(init_pts, deadline - 10.0, max_iters=5000)
        e_gd = _precise_energy(gd_result)
        
        if e_gd < best_energy:
            best_energy = e_gd
            best_pts = gd_result.copy()
            
    # Phase 2: BFGS polish on the best candidate
    if best_pts is not None and time.time() < deadline - 5.0:
        bfgs_result = _bfgs_optimize(best_pts, deadline - 2.0, max_iter=2000)
        e_bfgs = _precise_energy(bfgs_result)
        
        if e_bfgs < best_energy:
            best_energy = e_bfgs
            best_pts = bfgs_result.copy()
    
    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    # Validate
    if not _valid(best_pts):
        # Return a valid fallback
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _normalize(best_pts)
    
    return best_pts
