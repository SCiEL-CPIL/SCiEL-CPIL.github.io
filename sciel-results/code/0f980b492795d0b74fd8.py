"""Logarithmic energy minimization for N points on the unit sphere.
Combined approach: diverse initializations + gradient-based optimization
+ simulated annealing for escaping local minima + final polishing."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 150.0
_SAFETY = 15.0
_START = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)

def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n

def _energy(pts):
    """Fast energy computation using dot products."""
    n = pts.shape[0]
    dots = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))

def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space)."""
    n = pts.shape[0]
    dots = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    
    iu = np.triu_indices(n, k=1)
    dist = np.sqrt(dist_sq[iu])
    energy = float(-np.sum(np.log(dist)))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # Gradient: dE/dx_i = sum_j W_ij*(x_j - x_i) = (W @ pts)_i - row_sums_i * x_i
    grad = W @ pts - W.sum(axis=1, keepdims=True) * pts
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

def _optimize(pts, max_iter=2000, ftol=1e-14, gtol=1e-10):
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
                'maxfun': max_iter * 10,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts

def _polish_gradient_descent(pts, max_iter=5000, tol=1e-14):
    """Custom gradient descent with adaptive step size for final polish."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_e = _energy(pts)
    step = 0.1
    
    for it in range(max_iter):
        e, g = _energy_grad(best_pts)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        candidate = best_pts - step * g
        candidate = _project(candidate)
        cand_e = _energy(candidate)
        if cand_e < best_e:
            best_e = cand_e
            best_pts = candidate
            step = min(step * 1.1, 0.5)
        else:
            step *= 0.7
            if step < 1e-8:
                break
    return best_pts

def _simulated_annealing(pts, rng, start_temp=0.5, end_temp=0.01, n_iter=5000):
    """Simulated annealing to escape local minima.
    
    Uses geometric cooling schedule. Each iteration:
    1. Pick a random point
    2. Move it by a random tangent perturbation scaled by temperature
    3. Accept if energy decreases or with probability exp(-delta_E/T)
    """
    n = pts.shape[0]
    current_pts = pts.copy()
    current_e = _energy(current_pts)
    best_pts = current_pts.copy()
    best_e = current_e
    
    # Precompute number of iterations per temperature step
    n_steps = n_iter
    temps = np.geomspace(start_temp, end_temp, n_steps)
    
    for i, temp in enumerate(temps):
        # Pick random point
        idx = rng.integers(0, n)
        
        # Generate random tangent perturbation
        pert = rng.standard_normal(3)
        # Project to tangent space at current_pts[idx]
        pert = pert - np.dot(pert, current_pts[idx]) * current_pts[idx]
        pert_norm = np.linalg.norm(pert)
        if pert_norm < 1e-12:
            continue
        pert = pert / pert_norm
        
        # Scale perturbation by temperature
        step_size = temp * 0.1
        candidate = current_pts.copy()
        candidate[idx] = current_pts[idx] + step_size * pert
        candidate[idx] = _project(candidate[idx].reshape(1, 3))[0]
        
        cand_e = _energy(candidate)
        delta_e = cand_e - current_e
        
        # Accept if better or with Boltzmann probability
        if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
            current_pts = candidate
            current_e = cand_e
            if current_e < best_e:
                best_e = current_e
                best_pts = current_pts.copy()
    
    return best_pts, best_e

def _nudge_and_reoptimize(pts, rng, n_nudges=5, nudge_scale=0.05):
    """Try small nudges to escape local minima, then re-optimize."""
    best_pts = pts.copy()
    best_e = _precise_energy(pts)
    
    for _ in range(n_nudges):
        nudged = pts + nudge_scale * rng.standard_normal((N, 3))
        nudged = _project(nudged)
        
        optimized = _optimize(nudged, max_iter=300, ftol=1e-12, gtol=1e-8)
        e = _precise_energy(optimized)
        if e < best_e:
            best_e = e
            best_pts = optimized
    
    return best_pts

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
    deadline = start_time + 115.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(10):
        jitter_scale = 0.02 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Random points with different seeds
    for s in range(10):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
    
    # Additional diverse starts: rotated Fibonacci
    for angle in [0.3, 0.7, 1.1, 1.5]:
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        rotated = base @ rot.T
        initializations.append(_project(rotated))
    
    best_pts = None
    best_energy = float('inf')
    
    # First pass: quick evaluation of all initializations
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        
        e_quick = _energy(init_pts)
        if e_quick < best_energy:
            best_energy = e_quick
            best_pts = init_pts.copy()
    
    # Second pass: optimize the top candidates
    if best_pts is not None and time.time() < deadline - 20.0:
        scored_inits = []
        for init_pts in initializations:
            e_quick = _energy(init_pts)
            scored_inits.append((e_quick, init_pts))
        scored_inits.sort(key=lambda x: x[0])
        
        # Optimize top 8 candidates
        top_candidates = scored_inits[:8]
        for e_quick, init_pts in top_candidates:
            if time.time() > deadline - 15.0:
                break
            
            optimized = _optimize(init_pts, max_iter=800, ftol=1e-13, gtol=1e-9)
            e_precise = _precise_energy(optimized)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = optimized
    
    # Final polish on the global best with L-BFGS-B
    if best_pts is not None and time.time() < deadline - 25.0:
        refined = _optimize(best_pts, max_iter=5000, ftol=1e-16, gtol=1e-14)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
    
    # Simulated annealing to escape local minima
    if best_pts is not None and time.time() < deadline - 15.0:
        sa_pts, sa_e = _simulated_annealing(
            best_pts, rng, 
            start_temp=0.3, end_temp=0.01, 
            n_iter=3000
        )
        # Polish the SA result
        sa_polished = _optimize(sa_pts, max_iter=1000, ftol=1e-14, gtol=1e-12)
        e_precise = _precise_energy(sa_polished)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = sa_polished
    
    # Nudge and re-optimize to escape local minima
    if best_pts is not None and time.time() < deadline - 8.0:
        nudged_best = _nudge_and_reoptimize(best_pts, rng, n_nudges=3, nudge_scale=0.03)
        e_precise = _precise_energy(nudged_best)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = nudged_best
    
    # Custom gradient descent polish for extra precision
    if best_pts is not None and time.time() < deadline - 3.0:
        polished = _polish_gradient_descent(best_pts, max_iter=5000, tol=1e-15)
        e_precise = _precise_energy(polished)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = polished
    
    # One more L-BFGS-B pass after gradient descent polish
    if best_pts is not None and time.time() < deadline - 1.0:
        final_refined = _optimize(best_pts, max_iter=2000, ftol=1e-16, gtol=1e-15)
        e_precise = _precise_energy(final_refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = final_refined
    
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
