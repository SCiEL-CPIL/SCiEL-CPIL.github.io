"""Logarithmic energy minimization for N points on the unit sphere.
Stronger approach: correct gradient sign, diverse initializations,
multi-phase optimization with tangent-space L-BFGS-B and final polish."""
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
    dE/dx_i = sum_{j != i} (x_i - x_j) / d_ij^2   (this is the NEGATIVE gradient of -E)
    
    Wait, let me be careful:
    E = -sum_{i<j} log(d_ij)
    dE/dx_i = -sum_{j!=i} (1/d_ij) * d(d_ij)/dx_i
    d(d_ij)/dx_i = (x_i - x_j)/d_ij
    So dE/dx_i = -sum_{j!=i} (x_i - x_j) / d_ij^2
    
    We want to MINIMIZE E, so we follow -dE/dx_i = sum_{j!=i} (x_i - x_j) / d_ij^2
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
    
    # dE/dx_i = -sum_{j!=i} (x_i - x_j) / d_ij^2 = -(row_sums[i]*x_i - sum_j W[i,j]*x_j)
    # So -dE/dx_i = row_sums[i]*x_i - sum_j W[i,j]*x_j  (this is the direction to descend E)
    row_sums = W.sum(axis=1, keepdims=True)
    grad_E = row_sums * pts - W @ pts  # This is -dE/dx_i, i.e., the descent direction for E
    # Actually: dE/dx_i = -grad_E, so to minimize E we go along +grad_E
    
    # Project onto tangent space
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


def _sobol_like(n, seed=0):
    rng = np.random.default_rng(seed)
    points = []
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(n):
        t = (i + 0.5) / n
        z = 1.0 - 2.0 * t
        angle = 2.0 * np.pi * i * (1.0 + 1.0 / golden_ratio) + rng.standard_normal() * 0.05
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
        points.append([r * np.cos(angle), r * np.sin(angle), z])
    return _project(np.array(points))


def _optimize_lbfgs(pts, max_iter=5000, ftol=1e-15, gtol=1e-12):
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
    step = 0.05
    
    for it in range(max_iter):
        e, g = _energy_grad(best_pts)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        candidate = best_pts + step * g  # g is already the descent direction for E
        candidate = _project(candidate)
        cand_e = _energy(candidate)
        if cand_e < best_e:
            best_e = cand_e
            best_pts = candidate
            step = min(step * 1.05, 0.5)
        else:
            step *= 0.7
            if step < 1e-10:
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
    deadline = start_time + 135.0  # 15s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(15):
        jitter_scale = 0.01 + 0.008 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Random points with different seeds
    for s in range(15):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
    
    # Sobol-like points
    for s in range(8):
        initializations.append(_sobol_like(N, seed=s))
    
    # Rotated Fibonacci
    for angle in [0.3, 0.7, 1.1, 1.5, 2.0, 2.5]:
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        rotated = base @ rot.T
        initializations.append(_project(rotated))
    
    # Additional: random rotations of base
    for s in range(8):
        rng_r = np.random.default_rng(200 + s)
        v = rng_r.standard_normal((3, 3))
        q, _ = np.linalg.qr(v)
        rotated = base @ q.T
        initializations.append(_project(rotated))
    
    best_pts = None
    best_energy = float('inf')
    
    # First pass: quick evaluation of all initializations
    for init_pts in initializations:
        if time.time() > deadline - 60.0:
            break
        e_quick = _energy(init_pts)
        if e_quick < best_energy:
            best_energy = e_quick
            best_pts = init_pts.copy()
    
    # Sort initializations by energy and optimize top candidates
    if best_pts is not None:
        scored_inits = []
        for init_pts in initializations:
            if time.time() > deadline - 55.0:
                break
            e_quick = _energy(init_pts)
            scored_inits.append((e_quick, init_pts))
        scored_inits.sort(key=lambda x: x[0])
        
        # Optimize top 15 candidates with L-BFGS-B
        top_candidates = scored_inits[:15]
        for e_quick, init_pts in top_candidates:
            if time.time() > deadline - 30.0:
                break
            
            optimized = _optimize_lbfgs(init_pts, max_iter=3000, ftol=1e-14, gtol=1e-10)
            e_precise = _precise_energy(optimized)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = optimized
    
    # Second round: re-optimize the best with more iterations
    if best_pts is not None and time.time() < deadline - 15.0:
        refined = _optimize_lbfgs(best_pts, max_iter=10000, ftol=1e-16, gtol=1e-13)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
    
    # Gradient descent polish
    if best_pts is not None and time.time() < deadline - 5.0:
        polished = _gradient_descent_polish(best_pts, max_iter=50000, tol=1e-15)
        e_precise = _precise_energy(polished)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = polished
    
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
