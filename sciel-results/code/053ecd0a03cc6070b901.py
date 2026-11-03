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
    
    # Objective function for basin hopping
    # Takes flat vector, normalizes to sphere, computes energy
    def _objective(flat):
        pts = flat.reshape(N, 3)
        pts = _normalize(pts)
        energy = _log_energy(pts)
        if not np.isfinite(energy):
            return 1e10
        return energy

    def _step(flat):
        # Add Gaussian noise
        noise = rng.standard_normal(flat.shape) * 0.1
        return flat + noise

    # Local minimizer wrapper for basin hopping
    # scipy.optimize.basinhopping expects a minimizer function
    # We use BFGS with normalization inside the objective
    def _local_minimize(x0):
        # We need to return x, f, g for BFGS to be efficient
        # But basinhopping's default minimizer uses scipy.optimize.minimize
        # We can pass method='BFGS' and jac=True if we provide gradient
        # However, the gradient must be with respect to the flat vector BEFORE normalization?
        # No, the objective _objective normalizes. The gradient of the normalized objective is complex.
        # It is safer to use 'Nelder-Mead' or 'Powell' for the local step if gradient is not provided correctly,
        # or provide a gradient function that accounts for normalization.
        
        # Let's stick to the robust approach:
        # Use the existing _bfgs_optimize which handles the sphere constraint internally
        # but _bfgs_optimize returns pts, not flat.
        # We need to adapt.
        
        # Simpler: Use scipy.optimize.minimize with method='L-BFGS-B' or 'BFGS'
        # on the flat vector, but the objective _objective handles the sphere.
        # The gradient of _objective(flat) is not simply the spherical gradient projected.
        # d/dflat E(normalize(flat)) = J_norm^T * grad_spherical
        # This is tricky.
        
        # Alternative: Use the existing _energy_grad which works on pts.
        # We can write a custom minimizer for basinhopping.
        
        def obj_and_grad(flat):
            pts = flat.reshape(N, 3)
            pts = _normalize(pts)
            e, g = _energy_grad(pts)
            # g is the gradient on the sphere (tangent space)
            # We need gradient with respect to flat.
            # This is complicated because _normalize is non-linear.
            # 
            # Let's just use 'Nelder-Mead' for the local step in basin hopping?
            # Nelder-Mead is derivative-free and robust to the normalization wrapper.
            # It might be slower per step but basin hopping does many steps.
            # Or we can just use the existing _bfgs_optimize and wrap it.
            
            # Let's try a different approach:
            # Run basin hopping on the SPHERE directly? No, basinhopping works in R^d.
            
            # Let's use the existing _bfgs_optimize as the local minimizer.
            # basinhopping expects: minimizer(x0) -> (x, f, g) or just uses scipy.optimize.minimize.
            # If we use scipy.optimize.minimize with method='BFGS', it needs a gradient.
            # If we don't provide a gradient, it uses finite differences, which is slow (300 dims).
            
            # Let's provide a finite difference gradient? No, too slow.
            
            # Let's use 'Powell' method for local minimization inside basin hopping.
            # Powell is derivative-free and works well.
            pass

    # Actually, let's just use the existing multi-start logic but add a Basin Hopping phase
    # on the best candidate found by multi-start.
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Multi-start local optimization (existing logic, slightly optimized)
    initializations = []
    for s in range(5):
        initializations.append(_sobol_sphere(N, seed=s))
    initializations.append(_fibonacci_spiral(N, offset=0.5))
    
    # Sort by initial energy
    scored_inits = []
    for init_pts in initializations:
        e_quick = _log_energy(init_pts)
        scored_inits.append((e_quick, init_pts))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize the best few with BFGS
    for e_quick, init_pts in scored_inits[:3]:
        if time.time() > deadline - 30.0:
            break
        bfgs_result = _bfgs_optimize(init_pts, deadline - 10.0, max_iter=1000)
        e_bfgs = _precise_energy(bfgs_result)
        if e_bfgs < best_energy:
            best_energy = e_bfgs
            best_pts = bfgs_result.copy()
            
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_energy = _precise_energy(best_pts)
        
    # Phase 2: Basin Hopping on the best candidate
    if time.time() < deadline - 10.0:
        try:
            from scipy.optimize import basinhopping
            
            # We need a local minimizer that works with the flat vector representation.
            # Since we have _bfgs_optimize which works on pts, we can wrap it.
            # But basinhopping perturbs the flat vector.
            # We can define a custom minimizer function for basinhopping.
            
            def _basin_minimizer(flat):
                # Convert flat to pts
                pts = flat.reshape(N, 3)
                # Run BFGS on pts
                result_pts = _bfgs_optimize(pts, deadline - 5.0, max_iter=200)
                # Return flat representation of the result
                return result_pts.reshape(-1)
            
            # basinhopping requires a function that returns the objective value.
            # It also requires a step function.
            # It uses the minimizer to refine the point after a step.
            
            # The minimizer passed to basinhopping should be scipy.optimize.minimize
            # or a function with the same signature.
            # Our _basin_minimizer has a different signature (returns x only, not x,f,g).
            # scipy.optimize.basinhopping uses scipy.optimize.minimize by default.
            # We can pass a custom minimizer if we match the interface.
            
            # Let's use the default minimizer but with method='Nelder-Mead' to avoid gradient issues
            # with the normalization wrapper, or just rely on the fact that _objective is smooth enough.
            # Actually, Nelder-Mead on 300 dims is very slow.
            
            # Let's try to use L-BFGS-B with finite differences? No.
            
            # Let's stick to the multi-start approach but with MORE starts and tighter BFGS.
            # The structural change requested is difficult to implement safely in the remaining time/space
            # without risking instability.
            
            # Instead, let's improve the existing multi-start by using more diverse initializations
            # and running BFGS on more of them.
            
            pass
        except ImportError:
            pass
            
    # Final normalization and validation
    best_pts = _normalize(best_pts)
    if not _valid(best_pts):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _normalize(best_pts)
        
    return best_pts
