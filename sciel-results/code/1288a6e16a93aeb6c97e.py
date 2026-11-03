"""Optimized logarithmic energy minimization for 100 points on the unit sphere.
Combines Riemannian gradient descent with BFGS polish and diverse low-discrepancy
initializations (Fibonacci spirals, Sobol sequences, random)."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
import time
import math

N = 100
DEADLINE = 140.0  # Safety margin before 150s kill


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _energy_and_grad(points):
    """Compute logarithmic energy and gradient for points on unit sphere.
    Vectorized: Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    """
    n = points.shape[0]
    # Compute pairwise squared distances
    sq_norms = np.einsum('ij,ij->i', points, points)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (points @ points.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dist_sq[iu]))
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # Using: sum_j W_ij * (x_i - x_j) = x_i * sum_j W_ij - sum_j W_ij * x_j
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ points
    raw_grad = sum_W * points - Wx  # This is sum (x_i - x_j)/d^2
    
    # Energy gradient is -raw_grad (since E = -sum log(d))
    grad = -raw_grad
    
    # Project onto tangent space: grad - (grad . x_i) x_i
    dot = np.sum(grad * points, axis=1, keepdims=True)
    grad = grad - dot * points
    
    return energy, grad


def _energy_only(points):
    """Just the energy value (no gradient)."""
    n = points.shape[0]
    sq_norms = np.einsum('ij,ij->i', points, points)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (points @ points.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    iu = np.triu_indices(n, k=1)
    return -0.5 * np.sum(np.log(dist_sq[iu]))


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


def _exponential_map(points, tangent_vecs):
    """Riemannian exponential map on the sphere."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-15, 1.0, norms)
    unit_tangent = tangent_vecs / norms_safe
    cos_t = np.cos(norms)
    sin_t = np.sin(norms)
    new_points = points * cos_t + unit_tangent * sin_t
    return _normalize(new_points)


def _riemannian_gradient_descent(points, deadline, max_iter=3000, lr_init=0.01):
    """Riemannian gradient descent with backtracking line search."""
    current = points.copy()
    best_e = _energy_only(current)
    best_pts = current.copy()
    lr = lr_init
    c1 = 0.1
    rho = 0.5
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_and_grad(current)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-12:
            break
        
        # Backtracking line search
        step = lr
        found = False
        for bt in range(15):
            tangent = -step * g
            new_pts = _exponential_map(current, tangent)
            new_e = _energy_only(new_pts)
            if new_e <= e + c1 * step * g_norm * g_norm:
                found = True
                break
            step *= rho
        
        if not found:
            step = max(lr * rho**15, 1e-15)
            tangent = -step * g
            new_pts = _exponential_map(current, tangent)
            new_e = _energy_only(new_pts)
        
        if new_e < best_e - 1e-14:
            current = new_pts
            best_e = new_e
            best_pts = current.copy()
            no_improve = 0
            lr = min(step * 1.3, 0.05)
        elif new_e < best_e:
            current = new_pts
            no_improve += 1
            lr = step
        else:
            lr = step * 0.5
            no_improve += 1
            if lr < 1e-14 or no_improve > 40:
                break
    
    return best_pts, best_e


def _fibonacci_spiral(n, offset=0.5):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _random_points(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)


def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence (low discrepancy)."""
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


def _bfgs_optimize(pts, deadline, max_iter=1000, ftol=1e-15, gtol=1e-12):
    """Run BFGS optimization from a starting point."""
    n = pts.shape[0]
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        e, g = _energy_and_grad(p)
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
        new_pts = _normalize(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _lbfgsb_optimize(pts, deadline, max_iter=500, ftol=1e-15, gtol=1e-12):
    """Run L-BFGS-B optimization from a starting point."""
    n = pts.shape[0]
    
    def objective(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        return _energy_only(p)
    
    def gradient(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        _, g = _energy_and_grad(p)
        return g.reshape(-1)
    
    try:
        x0 = pts.reshape(-1)
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 30
            }
        )
        new_pts = _normalize(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # 2. Sobol sequences (low discrepancy)
    for s in range(15):
        initializations.append(_sobol_sphere(N, seed=s))
    
    # 3. Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(8):
        jitter_scale = 0.05 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_normalize(jittered))
    
    # 4. Random points with different seeds
    for s in range(10):
        initializations.append(_random_points(N, np.random.default_rng(100 + s)))
    
    # 5. Anti-symmetric pairs
    half = N // 2
    pts_half = _random_points(half, rng)
    pts_full = np.vstack([pts_half, -pts_half])
    if N % 2 == 1:
        pts_full = np.vstack([pts_full, np.array([[0, 0, 1.0]])])
    initializations.append(_normalize(pts_full))
    
    # Score all initializations
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 25.0:
            break
        e_quick = _energy_only(init_pts)
        scored_inits.append((e_quick, init_pts))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize top candidates
    best_points = scored_inits[0][1]
    best_energy = scored_inits[0][0]
    
    # Phase 1: Riemannian gradient descent for top candidates
    top_candidates = scored_inits[:25]
    
    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 15.0:
            break
        
        try:
            pts, e = _riemannian_gradient_descent(init_pts, deadline - 12.0, max_iter=1500)
            if e < best_energy:
                best_energy = e
                best_points = pts.copy()
        except Exception:
            continue
    
    # Phase 2: BFGS polish on best result
    if time.time() < deadline - 8.0:
        try:
            pts = _bfgs_optimize(best_points, deadline - 5.0, max_iter=500, ftol=1e-15, gtol=1e-12)
            e = _energy_only(pts)
            if e < best_energy:
                best_energy = e
                best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B final polish with tight tolerance
    if time.time() < deadline - 3.0:
        try:
            pts = _lbfgsb_optimize(best_points, deadline - 2.0, max_iter=300, ftol=1e-16, gtol=1e-14)
            e = _energy_only(pts)
            if e < best_energy:
                best_energy = e
                best_points = pts.copy()
        except Exception:
            pass
    
    # Ensure valid output
    best_points = _normalize(best_points)
    
    # Check for any points too close together
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                best_points[j] = best_points[j] + 1e-8 * rng.standard_normal(3)
                best_points = _normalize(best_points)
    
    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_spiral(N, offset=0.5)
    
    return best_points
