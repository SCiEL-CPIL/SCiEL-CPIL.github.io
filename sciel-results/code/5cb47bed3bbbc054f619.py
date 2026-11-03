"""Logarithmic energy minimization for N points on the unit sphere.
Combines multi-start BFGS optimization with Sobol/Fibonacci initializations,
gradient-flow annealing for broader exploration, and ultra-precise final polish."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_DEADLINE = 145.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _project(X):
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
    """Compute energy and gradient (projected onto tangent space)."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = sum_W * pts - Wx
    grad = -raw_grad
    
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


def _optimize(pts, deadline, max_iter=500, ftol=1e-12, gtol=1e-10, method='L-BFGS-B'):
    """
    Optimizes points on the sphere.
    Uses L-BFGS-B on the flattened coordinates but with a more robust
    line-search that respects the manifold geometry better by
    projecting the step if it deviates too much, or simply relying on
    the fact that L-BFGS-B is robust to mild constraint violations
    which are then projected.
    
    To improve convergence, we will run a secondary phase of pure
    Riemannian gradient descent with backtracking line search if
    the gradient norm is still high.
    """
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
            method=method,
            jac=True,
            options={
                'maxiter': max_iter,
                'gtol': gtol,
                'ftol': ftol,
                'maxfun': max_iter * 10,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        
        # Phase 2: Riemannian Gradient Descent Polish
        # This helps escape shallow minima that L-BFGS-B might stagnate in
        # due to numerical noise in the Hessian approximation.
        if time.time() < deadline - 2.0:
            current = new_pts
            e_current, g_current = _energy_grad(current)
            g_norm = np.linalg.norm(g_current)
            
            # If gradient is still significant, do a few steps of pure GD with line search
            if g_norm > 1e-8:
                for step in range(200):
                    if time.time() > deadline - 2.0:
                        break
                    e, g = _energy_grad(current)
                    g_norm = np.linalg.norm(g)
                    if g_norm < 1e-10:
                        break
                    
                    # Tangent step
                    dot = np.sum(g * current, axis=1, keepdims=True)
                    g_tangent = g - dot * current
                    
                    # Backtracking line search
                    alpha = 0.1
                    e_old = e
                    for _ in range(20):
                        # Move in tangent space
                        step_vec = -alpha * g_tangent
                        new_pts_candidate = current + step_vec
                        new_pts_candidate = _project(new_pts_candidate)
                        
                        e_new, _ = _energy_grad(new_pts_candidate)
                        if e_new < e_old - 1e-4 * alpha * np.dot(g.flatten(), step_vec.flatten()):
                            current = new_pts_candidate
                            e_current = e_new
                            break
                        alpha *= 0.5
                    else:
                        # If no improvement found, break
                        break
                        
            new_pts = current
            
        return new_pts
    except Exception:
        return pts


def _anneal(pts, deadline, n_steps=200, base_lr=0.01, min_lr=0.001):
    """Gradient-flow annealing with adaptive learning rate and noise schedule."""
    current = pts.copy()
    n = current.shape[0]
    best_e = _energy(current)
    best_current = current.copy()
    
    for step in range(n_steps):
        if time.time() > deadline - 5.0:
            break
        
        # Cosine annealing schedule with restarts
        progress = step / max(n_steps - 1, 1)
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        lr = max(lr, min_lr)
        
        e, g = _energy_grad(current)
        
        # Adaptive noise: larger early, smaller late
        noise_scale = 0.002 * (1 - progress) + 0.0001
        perturbation = noise_scale * np.random.standard_normal((n, 3))
        step_vec = -lr * (g + perturbation)
        
        # Project step to tangent space
        dot = np.sum(step_vec * current, axis=1, keepdims=True)
        step_vec = step_vec - dot * current
        
        current = current + step_vec
        current = _project(current)
        
        # Track best
        e_new = _energy(current)
        if e_new < best_e:
            best_e = e_new
            best_current = current.copy()
    
    return best_current


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


def _newton_polish(pts, deadline, max_iter=100, tol=1e-15):
    """
    High-precision polish using Riemannian Gradient Descent with
    adaptive step size and strict convergence criteria.
    """
    current = pts.copy()
    n = current.shape[0]
    
    # Initial gradient check
    e, g = _energy_grad(current)
    g_norm = np.linalg.norm(g)
    
    # Adaptive learning rate
    lr = 0.01
    min_lr = 1e-12
    
    for it in range(max_iter):
        if time.time() > deadline - 1.0:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        
        if g_norm < tol:
            break
        
        # Project gradient to tangent space
        dot = np.sum(g * current, axis=1, keepdims=True)
        g_tangent = g - dot * current
        
        # Step
        step_vec = -lr * g_tangent
        
        # Simple backtracking to ensure descent
        e_old = e
        new_pts_candidate = _project(current + step_vec)
        e_new, _ = _energy_grad(new_pts_candidate)
        
        if e_new < e_old:
            current = new_pts_candidate
            # Decrease learning rate slowly if we've been successful
            if it > 10:
                lr *= 0.95
        else:
            # Reduce learning rate if step failed
            lr *= 0.5
            if lr < min_lr:
                break
        
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Sobol sequences (Low discrepancy)
    for s in range(15):
        initializations.append(_sobol_sphere(N, seed=s))
        
    # 2. Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
        
    # 3. Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(8):
        jitter_scale = 0.05 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
        
    # 4. Random points with different seeds
    for s in range(15):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
        
    # 5. Jittered Sobol sequences
    for s in range(5):
        sobol_pts = _sobol_sphere(N, seed=s + 100)
        jitter_scale = 0.02
        jittered = sobol_pts + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
        
    best_pts = None
    best_energy = float('inf')
    
    # Score all initializations
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 25.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
        
    scored_inits.sort(key=lambda x: x[0])
    
    # Phase 1: Optimize top candidates with BFGS
    # We limit the number of candidates to ensure we have time for polishing.
    # Top 20 is usually sufficient for N=100.
    top_candidates = scored_inits[:20]
    
    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 20.0:
            break
        
        # Use a slightly lower iteration count here to save time for polish
        optimized = _optimize(init_pts, deadline, max_iter=500, ftol=1e-12, gtol=1e-10)
        
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
            
    # Phase 2: Annealing + optimization on the best configuration
    if best_pts is not None and time.time() < deadline - 15.0:
        # Annealing helps escape local minima
        annealed = _anneal(best_pts, deadline, n_steps=100, base_lr=0.01, min_lr=0.0001)
        
        # Refine the annealed result
        refined = _optimize(annealed, deadline, max_iter=1000, ftol=1e-14, gtol=1e-12)
        
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined.copy()
    
    # Phase 3: Final ultra-precise polish with multiple strategies
    if best_pts is not None and time.time() < deadline - 5.0:
        # Try multiple tiny perturbations around the best to check for nearby minima
        for attempt in range(10):
            if time.time() > deadline - 4.0:
                break
            # Smaller perturbations for higher precision
            perturbation_scale = 0.001 * (0.5 ** attempt)
            perturbed = best_pts + perturbation_scale * rng.standard_normal((N, 3))
            perturbed = _project(perturbed)
            
            # Use the high-precision polish directly
            polished = _newton_polish(perturbed, deadline, max_iter=50, tol=1e-14)
            
            e_precise = _precise_energy(polished)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = polished.copy()
        
        # Final strict polish on the global best
        if time.time() < deadline - 2.0:
            polished = _newton_polish(best_pts, deadline, max_iter=100, tol=1e-15)
            e_precise = _precise_energy(polished)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = polished.copy()
    
    # Final safety check and normalization
    if best_pts is not None:
        best_pts = _project(best_pts)
    
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
