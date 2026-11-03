"""Logarithmic energy minimization for N points on the unit sphere.
Search-based approach: generate many varied candidates and keep the best."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 150.0
_SAFETY = 15.0
_START = time.time()

def _spherical_rbf_init(n, seed=0):
    """Initialize points using a spherical RBF/GP-inspired approach.
    Place points to maximize minimum distance using a repulsive potential."""
    rng = np.random.default_rng(seed)
    # Start with Fibonacci spiral and iteratively move each point to maximize
    # its minimum distance to all others (greedy repulsion)
    pts = _fibonacci_spiral(n, offset=0.5)
    for it in range(3):
        for i in range(n):
            # Find direction of maximum repulsion from point i
            best_dir = None
            best_score = -np.inf
            # Try random perturbations
            for _ in range(50):
                pert = rng.standard_normal(3)
                cand = pts[i] + 0.1 * pert
                cand = _project(cand.reshape(1, 3))[0]
                # Score: sum of log distances to all other points
                dists = np.linalg.norm(pts - cand, axis=1)
                dists[i] = 1.0  # exclude self
                score = np.sum(np.log(dists))
                if score > best_score:
                    best_score = score
                    best_dir = cand
            if best_dir is not None:
                pts[i] = best_dir
    return _project(pts)


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space).
    
    E = -sum_{i<j} log(d_ij)
    dE/dx_i = sum_{j != i} W_ij * (x_j - x_i) = (W @ pts)_i - (sum_j W_ij) * x_i
    where W_ij = 1/d_ij^2
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
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

def _icosahedral_based(n, seed=0):
    """Generate points based on icosahedral symmetry with perturbation."""
    rng = np.random.default_rng(seed)
    # Start with a well-distributed set and add small random perturbations
    base = _fibonacci_spiral(n, offset=0.3)
    # Apply random rotations
    angles = rng.uniform(0, 2*np.pi, size=(3,))
    for axis, angle in enumerate(angles):
        c, s = np.cos(angle), np.sin(angle)
        if axis == 0:
            rot = np.array([[1,0,0],[0,c,-s],[0,s,c]])
        elif axis == 1:
            rot = np.array([[c,0,s],[0,1,0],[-s,0,c]])
        else:
            rot = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        base = base @ rot.T
    # Small random perturbation
    perturbed = base + 0.01 * rng.standard_normal((n, 3))
    return _project(perturbed)


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_like(n, seed=0):
    """Generate well-distributed points using a simple quasi-random approach."""
    rng = np.random.default_rng(seed)
    points = []
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(n):
        t = (i + 0.5) / n
        z = 1.0 - 2.0 * t
        angle = 2.0 * np.pi * i * (1.0 + 1.0 / golden_ratio) + rng.standard_normal() * 0.1
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
        points.append([r * np.cos(angle), r * np.sin(angle), z])
    return _project(np.array(points))


def _optimize(pts, deadline, max_iter=1000, ftol=1e-15, gtol=1e-12):
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
    step = 0.5
    
    for it in range(max_iter):
        e, g = _energy_grad(best_pts)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        # Adaptive step size with line search
        candidate = best_pts - step * g
        candidate = _project(candidate)
        cand_e = _energy(candidate)
        if cand_e < best_e:
            best_e = cand_e
            best_pts = candidate
            step = min(step * 1.2, 1.0)
        else:
            # Backtracking line search
            for _ in range(10):
                step *= 0.5
                candidate = best_pts - step * g
                candidate = _project(candidate)
                cand_e = _energy(candidate)
                if cand_e < best_e:
                    best_e = cand_e
                    best_pts = candidate
                    break
            else:
                if step < 1e-10:
                    break
    return best_pts

def _nudge_and_reoptimize(pts, rng, n_nudges=5, nudge_scale=0.05):
    """Try small nudges to escape local minima, then re-optimize."""
    best_pts = pts.copy()
    best_e = _energy(pts)
    
    for _ in range(n_nudges):
        # Apply small random perturbation
        nudged = pts + nudge_scale * rng.standard_normal((N, 3))
        nudged = _project(nudged)
        
        # Quick optimization
        optimized = _optimize(nudged, deadline=_DEADLINE - 5.0, max_iter=300, ftol=1e-12, gtol=1e-8)
        e = _energy(optimized)
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
    
    # Sobol-like points
    for s in range(5):
        initializations.append(_sobol_like(N, seed=s))
    
    # Additional diverse starts: rotated Fibonacci
    for angle in [0.3, 0.7, 1.1, 1.5]:
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        rotated = base @ rot.T
        initializations.append(_project(rotated))
    
    # RBF-inspired initializations
    for s in range(3):
        initializations.append(_spherical_rbf_init(N, seed=s))
    
    # Icosahedral-based with perturbation
    for s in range(5):
        initializations.append(_icosahedral_based(N, seed=s))
    
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
        # Sort initializations by quick energy and take top ones
        scored_inits = []
        for init_pts in initializations:
            e_quick = _energy(init_pts)
            scored_inits.append((e_quick, init_pts))
        scored_inits.sort(key=lambda x: x[0])
        
        # Optimize top 10 candidates
        top_candidates = scored_inits[:10]
        for e_quick, init_pts in top_candidates:
            if time.time() > deadline - 15.0:
                break
            
            optimized = _optimize(init_pts, deadline, max_iter=800, ftol=1e-13, gtol=1e-9)
            e_precise = _precise_energy(optimized)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = optimized
    
    # Final polish on the global best with L-BFGS-B
    if best_pts is not None and time.time() < deadline - 8.0:
        refined = _optimize(best_pts, deadline, max_iter=5000, ftol=1e-16, gtol=1e-14)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
    
    # Nudge and re-optimize to escape local minima
    if best_pts is not None and time.time() < deadline - 5.0:
        nudged_best = _nudge_and_reoptimize(best_pts, rng, n_nudges=5, nudge_scale=0.05)
        e_precise = _precise_energy(nudged_best)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = nudged_best
    
    # Custom gradient descent polish for extra precision
    if best_pts is not None and time.time() < deadline - 2.0:
        polished = _polish_gradient_descent(best_pts, max_iter=10000, tol=1e-15)
        e_precise = _precise_energy(polished)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = polished
    
    # One more L-BFGS-B pass after gradient descent polish
    if best_pts is not None and time.time() < deadline - 1.0:
        final_refined = _optimize(best_pts, deadline, max_iter=2000, ftol=1e-16, gtol=1e-15)
        e_precise = _precise_energy(final_refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = final_refined
    
    # Final high-precision polish: alternating L-BFGS-B and gradient descent
    if best_pts is not None and time.time() < deadline - 0.5:
        for round in range(3):
            if time.time() > deadline - 0.5:
                break
            refined2 = _optimize(best_pts, deadline, max_iter=2000, ftol=1e-16, gtol=1e-15)
            e_precise = _precise_energy(refined2)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = refined2
            polished2 = _polish_gradient_descent(best_pts, max_iter=5000, tol=1e-15)
            e_precise = _precise_energy(polished2)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = polished2
    
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
