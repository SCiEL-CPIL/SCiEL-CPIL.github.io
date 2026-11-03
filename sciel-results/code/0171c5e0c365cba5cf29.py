"""Logarithmic energy minimization for N points on the unit sphere.
Combined approach: Sobol initialization, repulsive pre-optimization,
multi-phase L-BFGS-B with tangent-space projection, and gradient polish."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)."""
    n = pts.shape[0]
    sq = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """
    Energy E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    dE/dx_i = -sum_{j!=i} (x_i - x_j) / d_ij^2
    """
    n = pts.shape[0]
    sq = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    row_sums = W.sum(axis=1, keepdims=True)
    grad_E = -(row_sums * pts - W @ pts)
    
    # Project onto tangent space
    dot = np.sum(grad_E * pts, axis=1, keepdims=True)
    grad = grad_E - dot * pts
    
    return energy, grad


def _repulsive_energy_grad(pts, beta=2.0):
    """Repulsive energy: sum_{i<j} 1/d_ij^beta for pre-optimization."""
    n = pts.shape[0]
    sq = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = float(np.sum(1.0 / dist[iu] ** beta))
    
    W = beta / dist_sq ** (beta / 2.0)
    np.fill_diagonal(W, 0.0)
    
    row_sums = W.sum(axis=1, keepdims=True)
    grad_E = -(row_sums * pts - W @ pts)
    
    dot = np.sum(grad_E * pts, axis=1, keepdims=True)
    grad = grad_E - dot * pts
    
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


def _sobol_init(seed=0):
    """Sobol sequence initialization for better space coverage."""
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
        pts = sampler.random(N)
        z = 2.0 * pts[:, 2] - 1.0
        phi = 2.0 * math.pi * pts[:, 0]
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        result = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return _project(result)
    except Exception:
        return _fibonacci_spiral(N, offset=0.5)


def _optimize_lbfgs(pts, max_iter=5000, ftol=1e-15, gtol=1e-12, use_repulsive=False, beta=2.0):
    """Run L-BFGS-B optimization from a starting point."""
    n = pts.shape[0]
    
    if use_repulsive:
        def objective_and_grad(flat):
            p = flat.reshape(n, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            p = p / norms
            e, g = _repulsive_energy_grad(p, beta=beta)
            return e, g.reshape(-1)
    else:
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
                'maxfun': max_iter * 20,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _gradient_descent_polish(pts, max_iter=20000, tol=1e-14):
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
        
        alpha = step
        for _ in range(30):
            candidate = best_pts - alpha * g
            candidate = _project(candidate)
            cand_e = _energy(candidate)
            if cand_e < best_e - 1e-4 * alpha * g_norm * g_norm:
                break
            alpha *= 0.5
        
        if cand_e < best_e:
            best_e = cand_e
            best_pts = candidate
            step = min(alpha * 1.5, 0.5)
        else:
            step *= 0.5
            if step < 1e-12:
                break
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
    deadline = start_time + 145.0
    
    rng = np.random.default_rng(42)
    
    initializations = []
    
    # Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Sobol initializations - better space coverage
    for s in range(12):
        initializations.append(_sobol_init(seed=s))
    
    # Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(10):
        jitter_scale = 0.01 + 0.008 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Random points with different seeds
    for s in range(10):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
    
    # Rotated Fibonacci
    for angle in [0.3, 0.7, 1.1, 1.5, 2.0, 2.5]:
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        rotated = base @ rot.T
        initializations.append(_project(rotated))
    
    best_pts = None
    best_energy = float('inf')
    
    # Quick evaluation of all initializations
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 90.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Phase 1: Repulsive pre-optimization on top candidates to spread points
    top_candidates = scored_inits[:15]
    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 70.0:
            break
        
        # Repulsive pre-optimization to avoid clustering
        pre_opt = _optimize_lbfgs(init_pts, max_iter=2000, ftol=1e-10, gtol=1e-6, 
                                   use_repulsive=True, beta=2.0)
        
        # Then log-energy optimization
        optimized = _optimize_lbfgs(pre_opt, max_iter=3000, ftol=1e-12, gtol=1e-8)
        optimized = _optimize_lbfgs(optimized, max_iter=3000, ftol=1e-14, gtol=1e-10)
        
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized
    
    # Phase 2: Direct optimization on top candidates (without repulsive)
    for e_quick, init_pts in scored_inits[:10]:
        if time.time() > deadline - 40.0:
            break
        
        optimized = _optimize_lbfgs(init_pts, max_iter=3000, ftol=1e-12, gtol=1e-8)
        optimized = _optimize_lbfgs(optimized, max_iter=3000, ftol=1e-14, gtol=1e-10)
        
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized
    
    # Phase 3: Re-optimize the best with more iterations
    if best_pts is not None and time.time() < deadline - 30.0:
        refined = _optimize_lbfgs(best_pts, max_iter=20000, ftol=1e-16, gtol=1e-12)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
        
        if time.time() < deadline - 25.0:
            refined2 = _optimize_lbfgs(best_pts, max_iter=10000, ftol=1e-16, gtol=1e-13)
            e_precise = _precise_energy(refined2)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = refined2
    
    # Phase 4: Gradient descent polish
    if best_pts is not None and time.time() < deadline - 15.0:
        polished = _gradient_descent_polish(best_pts, max_iter=50000, tol=1e-15)
        e_precise = _precise_energy(polished)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = polished
    
    # Phase 5: Final L-BFGS polish after gradient descent
    if best_pts is not None and time.time() < deadline - 8.0:
        final_refined = _optimize_lbfgs(best_pts, max_iter=15000, ftol=1e-16, gtol=1e-14)
        e_precise = _precise_energy(final_refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = final_refined
    
    # Phase 6: Additional polish if time permits
    if best_pts is not None and time.time() < deadline - 4.0:
        extra_polish = _gradient_descent_polish(best_pts, max_iter=30000, tol=1e-16)
        e_precise = _precise_energy(extra_polish)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = extra_polish
    
    # Ultra-fine final L-BFGS
    if best_pts is not None and time.time() < deadline - 1.5:
        ultra_final = _optimize_lbfgs(best_pts, max_iter=20000, ftol=1e-17, gtol=1e-15)
        e_precise = _precise_energy(ultra_final)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = ultra_final
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    best_pts = _project(best_pts)
    
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
