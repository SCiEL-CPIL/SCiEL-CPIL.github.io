"""Logarithmic energy minimization for N points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 150.0          # hard kill from evaluator
_SAFETY = 15.0             # safety margin -> work until ~135s
_START = time.time()


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _fast_energy_and_grad(pts):
    """Compute logarithmic energy and gradient using vectorized operations.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    Returns (energy, gradient) where gradient has shape (N, 3).
    The gradient is projected onto the tangent space of the sphere.
    """
    n = pts.shape[0]
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    
    # Avoid division by zero
    np.fill_diagonal(dist, 1.0)
    
    # Energy: -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = -sum_{j!=i} (x_i - x_j) / dist_ij^2
    # = sum_{j!=i} (x_j - x_i) / dist_ij^2
    # Note: dist^2 = sum of diff^2
    dist_sq = dist * dist
    np.fill_diagonal(dist_sq, 1.0)
    
    # Weight matrix: w_ij = 1/dist_ij^2 for i != j
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = sum_j W_ij * (x_j - x_i) = (W @ pts)_i - W[i,:].sum() * x_i
    grad = W @ pts - W.sum(axis=1, keepdims=True) * pts
    
    # Project gradient onto tangent space: grad_tangent = grad - (grad . x_i) * x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return float(energy), grad


def _fast_energy_only(pts):
    """Compute just the logarithmic energy (faster, no gradient)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


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


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_sphere(n, seed=0):
    """Generate quasi-random points on the sphere using a simple low-discrepancy sequence."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    pts = sampler.random(n)
    # Transform to sphere: use the fact that uniform in cube -> uniform on sphere via normalization
    v = pts - 0.5  # center to [-0.5, 0.5]
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return v / norms


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    # min pairwise distance
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def _optimize_from_start(pts, deadline):
    """Run L-BFGS-B optimization from a starting point, respecting deadline."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        # Normalize to sphere
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        
        e, g = _fast_energy_and_grad(p)
        g_flat = g.reshape(-1)
        return e, g_flat
    
    # Phase 1: coarse optimization with moderate tolerances
    if time.time() < deadline - 5.0:
        try:
            x0 = pts.reshape(-1)
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 200,
                    'maxfun': 1000,
                    'ftol': 1e-8,
                    'gtol': 1e-5,
                    'maxls': 20,
                }
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    # Phase 2: medium refinement
    if time.time() < deadline - 5.0:
        try:
            x0 = best_pts.reshape(-1)
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 400,
                    'maxfun': 2000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 30,
                }
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    # Phase 3: fine refinement
    if time.time() < deadline - 5.0:
        try:
            x0 = best_pts.reshape(-1)
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 600,
                    'maxfun': 4000,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 40,
                }
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spiral with various offsets
    initializations.append(_fibonacci_spiral(N, offset=0.5))
    initializations.append(_fibonacci_spiral(N, offset=0.3))
    initializations.append(_fibonacci_spiral(N, offset=0.7))
    
    # Jittered Fibonacci spirals with varying amplitudes
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(4):
        amp = 0.03 + 0.02 * s
        jittered = base + amp * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Sobol quasi-random points (better coverage than pure random)
    for s in range(5):
        initializations.append(_sobol_sphere(N, seed=s * 13))
    
    # Random points with different seeds
    for s in range(5):
        initializations.append(_random_sphere(100 + s * 7))
    
    # Points from icosahedral-like structure (12 vertices) replicated with small perturbations
    ico = np.array([
        [0, 1, 0], [0, -1, 0],
        [1, 0, 0], [-1, 0, 0],
        [0, 0, 1], [0, 0, -1],
        [0.525731, 0.850651, 0], [-0.525731, 0.850651, 0],
        [0.525731, -0.850651, 0], [-0.525731, -0.850651, 0],
        [0.525731, 0, 0.850651], [-0.525731, 0, 0.850651],
        [0.525731, 0, -0.850651], [-0.525731, 0, -0.850651],
    ])
    ico = _project(ico)
    # Create 100 points by clustering around icosahedral vertices
    ico_pts = np.repeat(ico, 100 // len(ico) + 1, axis=0)[:N]
    ico_pts = ico_pts + 0.02 * rng.standard_normal((N, 3))
    initializations.append(_project(ico_pts))
    
    best_pts = None
    best_energy = float('inf')
    
    # First pass: try all initializations with optimization
    for init_pts in initializations:
        if time.time() > deadline - 20.0:  # reserve 20s for final polishing
            break
        
        # Optimize from this start
        optimized, e_precise = _optimize_from_start(init_pts, deadline - 15.0)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized
    
    # Final polish on the global best with multiple aggressive refinement passes
    if best_pts is not None:
        n = best_pts.shape[0]
        
        def objective_and_grad(flat):
            p = flat.reshape(n, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            p = p / norms
            e, g = _fast_energy_and_grad(p)
            g_flat = g.reshape(-1)
            return e, g_flat
        
        # Multiple polishing passes with progressively tighter tolerances
        polish_configs = [
            {'maxiter': 800, 'maxfun': 6000, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 50},
            {'maxiter': 1000, 'maxfun': 8000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 60},
            {'maxiter': 1200, 'maxfun': 10000, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 70},
            {'maxiter': 1500, 'maxfun': 12000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 80},
            {'maxiter': 2000, 'maxfun': 15000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100},
        ]
        
        for cfg in polish_configs:
            if time.time() > deadline - 3.0:
                break
            try:
                x0 = best_pts.reshape(-1)
                result = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options=cfg
                )
                new_pts = _project(result.x.reshape(n, 3))
                e_precise = _precise_energy(new_pts)
                if e_precise < best_energy:
                    best_energy = e_precise
                    best_pts = new_pts
            except Exception:
                pass
    
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
