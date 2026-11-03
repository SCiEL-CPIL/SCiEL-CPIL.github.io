"""
Logarithmic energy minimization for N points on the unit sphere.
Combines vectorized gradient descent with multi-phase L-BFGS-B polish and
perturbation-based escape from local minima. Uses Sobol and Fibonacci
initializations for diverse starts.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_IU = np.triu_indices(N, k=1)

def _fast_energy_and_grad(pts):
    """Compute logarithmic energy and gradient using vectorized operations."""
    n = pts.shape[0]
    dot_products = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot_products
    np.fill_diagonal(dist_sq, 1.0)
    np.maximum(dist_sq, 1e-15, out=dist_sq)
    
    energy = -0.5 * np.sum(np.log(dist_sq[_IU]))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    W_row_sum = W.sum(axis=1, keepdims=True)
    grad = W @ pts - W_row_sum * pts
    
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= dot * pts
    
    return float(energy), grad

def _fast_energy_only(pts):
    """Compute just the logarithmic energy (faster, no gradient)."""
    dot_products = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot_products
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    return float(-0.5 * np.sum(np.log(dist_sq[_IU])))

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

def _sobol_sphere(seed):
    """Generate points on the sphere using Sobol sequence."""
    sampler = qmc.Sobol(d=3, seed=seed)
    samples = sampler.random(N)
    samples = samples - 0.5
    return _project(samples)

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

def _gradient_flow(pts, rng, n_steps, lr_start, lr_end, deadline):
    """Gradient flow with adaptive learning rate and annealing."""
    n = pts.shape[0]
    current = pts.copy()
    best = pts.copy()
    best_energy = _fast_energy_only(pts)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        progress = step / max(n_steps - 1, 1)
        lr = lr_start * (lr_end / lr_start) ** progress
        
        e, g = _fast_energy_and_grad(current)
        
        # Adaptive learning rate: scale by inverse gradient magnitude
        g_norm = np.linalg.norm(g)
        if g_norm > 1e-12:
            lr = min(lr, 0.5 / g_norm)
        
        # Add small noise for exploration (annealed)
        noise_scale = 0.003 * (1.0 - progress) ** 1.5
        noise = rng.standard_normal((n, 3)) * noise_scale
        
        # Update in ambient space, then project back to sphere
        current = current - lr * g + noise
        current = _project(current)
        
        e_new = _fast_energy_only(current)
        if e_new < best_energy:
            best_energy = e_new
            best = current.copy()
    
    return best, best_energy

def _optimize_from_start(pts, deadline):
    """Run L-BFGS-B optimization from a starting point, respecting deadline."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        p /= norms
        
        e, g = _fast_energy_and_grad(p)
        return e, g.reshape(-1)
    
    # Phase 1: coarse optimization
    if time.time() < deadline - 3.0:
        try:
            result = minimize(
                objective_and_grad,
                pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20}
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    # Phase 2: medium refinement
    if time.time() < deadline - 3.0:
        try:
            result = minimize(
                objective_and_grad,
                best_pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'maxfun': 2500, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 30}
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    # Phase 3: fine refinement
    if time.time() < deadline - 2.0:
        try:
            result = minimize(
                objective_and_grad,
                best_pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 800, 'maxfun': 4000, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 40}
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    return best_pts, best_energy

def _final_polish(pts, deadline):
    """Final ultra-precise polish with tiny perturbations and multiple restarts."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        p /= norms
        e, g = _fast_energy_and_grad(p)
        return e, g.reshape(-1)
    
    rng = np.random.default_rng(12345)
    # More aggressive and varied perturbations
    perturbations = [0.0, 5e-4, 3e-4, 2e-4, 1.5e-4, 1e-4, 7e-5, 5e-5, 3e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]
    
    for scale in perturbations:
        if time.time() > deadline - 2.0:
            break
        
        # Try multiple restarts for each scale to escape shallow local minima
        n_restarts = 3 if scale > 0 else 1
        
        for restart in range(n_restarts):
            if time.time() > deadline - 2.0:
                break
            
            if scale == 0.0:
                start = best_pts
            else:
                start = _project(best_pts + rng.standard_normal((n, 3)) * scale)
            
            try:
                result = minimize(
                    objective_and_grad,
                    start.reshape(-1),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'maxfun': 2500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
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
    deadline = start_time + 135.0  # 15 second safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Sobol sequences (Low discrepancy)
    for s in range(10):
        initializations.append(_sobol_sphere(seed=s))
        
    # 2. Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
        
    # 3. Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(5):
        jitter_scale = 0.05 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
        
    # 4. Random points with different seeds
    for s in range(10):
        initializations.append(_random_sphere(100 + s))
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Gradient flow on top initializations
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 40.0:
            break
        e_quick = _fast_energy_only(init_pts)
        scored_inits.append((e_quick, init_pts))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize top candidates with gradient flow
    top_candidates = scored_inits[:10]
    
    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 30.0:
            break
        
        # Gradient flow first
        gd_result, e_gd = _gradient_flow(init_pts, rng, 3000, 0.1, 0.001, deadline - 10.0)
        e_precise = _precise_energy(gd_result)
        
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = gd_result.copy()
            
    # Phase 2: L-BFGS-B polish on the best candidate
    if best_pts is not None and time.time() < deadline - 15.0:
        bfgs_result, e_bfgs = _optimize_from_start(best_pts, deadline - 10.0)
        
        if e_bfgs < best_energy:
            best_energy = e_bfgs
            best_pts = bfgs_result.copy()
    
    # Phase 3: Final polish with perturbations
    if best_pts is not None and time.time() < deadline - 5.0:
        final_result, e_final = _final_polish(best_pts, deadline - 2.0)
        
        if e_final < best_energy:
            best_energy = e_final
            best_pts = final_result.copy()
    
    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization
    best_pts = _project(best_pts)
    
    # Validate
    if not _valid(best_pts):
        # Return a valid fallback
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
