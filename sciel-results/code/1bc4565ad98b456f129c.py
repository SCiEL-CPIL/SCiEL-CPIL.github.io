"""Optimized logarithmic energy minimization for 100 points on the sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)"""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    log_d = np.log(dist[iu])
    return -math.fsum(log_d)

def _gradient_log_energy(pts):
    """Tangent-space gradient of logarithmic energy, optimized with matrix ops."""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    sum_inv_d2 = inv_d2.sum(axis=1)
    sum_xj_inv_d2 = (pts.T @ inv_d2).T
    
    # grad of E = -sum_{i<j} log(d_ij) is: -sum_{j!=i} (x_i - x_j)/d^2
    grad = -(pts * sum_inv_d2[:, np.newaxis] - sum_xj_inv_d2)
    
    grad_dot_xi = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - grad_dot_xi * pts
    
    return grad_tan

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)

def _perturb_sphere(pts, rng, sigma=0.05):
    """Perturb points and renormalize."""
    noise = rng.standard_normal(pts.shape) * sigma
    new_pts = pts + noise
    return _project_to_sphere(new_pts)

def _icosahedron_subdivide(n, rng):
    """Generate points by subdividing an icosahedron."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = _project_to_sphere(verts)
    
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    ]
    
    pts = []
    for v in verts:
        pts.append(v)
    
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i+1) % 3]
            edge = (min(a,b), max(a,b))
            edges.add(edge)
    
    for a, b in edges:
        mid = (verts[a] + verts[b]) / 2.0
        pts.append(_project_to_sphere(mid.reshape(1,3))[0])
    
    for f in faces:
        center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        pts.append(_project_to_sphere(center.reshape(1,3))[0])
    
    pts = np.array(pts)
    if len(pts) > n:
        indices = rng.choice(len(pts), n, replace=False)
        pts = pts[indices]
    elif len(pts) < n:
        remaining = _random_sphere(n - len(pts), rng)
        pts = np.vstack([pts, remaining])
    
    return _project_to_sphere(pts)

def _golden_spiral_v2(n, offset=0.0):
    """Alternative golden spiral with offset for diversity."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    z = np.clip(z, -1.0, 1.0)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _riemannian_descent(pts, max_iter=500, tol=1e-10, deadline=None):
    """Riemannian gradient descent with backtracking line search on the sphere."""
    n = pts.shape[0]
    pts = _project_to_sphere(pts.copy())
    f = _log_energy(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _gradient_log_energy(pts)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Initial step size - more aggressive
        step = 1.0 / (1.0 + 0.5 * grad_norm)
        
        accepted = False
        for ls_iter in range(40):
            # Riemannian exponential map: move along tangent direction (negative gradient)
            new_pts = pts - step * grad
            new_pts = _project_to_sphere(new_pts)
            new_f = _log_energy(new_pts)
            
            if np.isfinite(new_f) and new_f < f:
                # Accept if it's a decrease (Armijo-like)
                if new_f < f - 1e-4 * step * grad_norm * grad_norm:
                    accepted = True
                    break
                # Accept any sufficient decrease
                if new_f < f - 1e-8:
                    accepted = True
                    break
            # Backtrack
            step *= 0.7
        
        if accepted:
            pts = new_pts
            f = new_f
        else:
            break
    
    return pts, f

def _optimize_from_start(pts0, time_budget, deadline):
    """Optimize from a starting configuration."""
    best_pts = _project_to_sphere(pts0.copy())
    best_energy = _log_energy(best_pts)
    
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        return _log_energy(pts)
    
    def gradient(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        g = _gradient_log_energy(pts)
        # For L-BFGS-B on unconstrained space, we use the Euclidean gradient
        # The projection constraint is handled by re-normalizing
        return g.reshape(-1)
    
    x0 = best_pts.reshape(-1)
    
    # First pass: L-BFGS-B with moderate tolerance
    try:
        res = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-8, 'maxfun': 1000, 'ftol': 1e-10}
        )
        if res.fun < best_energy:
            pts_new = _project_to_sphere(res.x.reshape(N, 3))
            e_new = _log_energy(pts_new)
            if e_new < best_energy:
                best_pts = pts_new
                best_energy = e_new
    except Exception:
        pass
    
    if time.time() > deadline - 5.0:
        return best_pts, best_energy
    
    # Second pass: Riemannian gradient descent with backtracking for robustness
    if time.time() < deadline - 8.0:
        try:
            pts_rd, e_rd = _riemannian_descent(best_pts, max_iter=500, tol=1e-12, deadline=deadline - 8.0)
            if e_rd < best_energy:
                best_pts = pts_rd
                best_energy = e_rd
        except Exception:
            pass
    
    if time.time() > deadline - 5.0:
        return best_pts, best_energy
    
    # Third pass: L-BFGS-B with tighter tolerance
    try:
        x1 = best_pts.reshape(-1)
        res2 = minimize(
            objective,
            x1,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 800, 'gtol': 1e-12, 'maxfun': 1500, 'ftol': 1e-14}
        )
        if res2.fun < best_energy:
            pts_new = _project_to_sphere(res2.x.reshape(N, 3))
            e_new = _log_energy(pts_new)
            if e_new < best_energy:
                best_pts = pts_new
                best_energy = e_new
    except Exception:
        pass
    
    # Fourth pass: Riemannian descent again for additional refinement
    if time.time() < deadline - 5.0:
        try:
            pts_rd2, e_rd2 = _riemannian_descent(best_pts, max_iter=400, tol=1e-13, deadline=deadline - 5.0)
            if e_rd2 < best_energy:
                best_pts = pts_rd2
                best_energy = e_rd2
        except Exception:
            pass
    
    # Fifth pass: very tight tolerance for final polish
    if time.time() < deadline - 3.0:
        try:
            x2 = best_pts.reshape(-1)
            res3 = minimize(
                objective,
                x2,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-14, 'maxfun': 2000, 'ftol': 1e-16}
            )
            if res3.fun < best_energy:
                pts_new = _project_to_sphere(res3.x.reshape(N, 3))
                e_new = _log_energy(pts_new)
                if e_new < best_energy:
                    best_pts = pts_new
                    best_energy = e_new
        except Exception:
            pass
    
    # Final ultra-tight polish
    if time.time() < deadline - 1.0:
        try:
            pts_final, e_final = _riemannian_descent(best_pts, max_iter=800, tol=1e-14, deadline=deadline - 1.0)
            if e_final < best_energy:
                best_pts = pts_final
                best_energy = e_final
        except Exception:
            pass
    
    return best_pts, best_energy

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_sphere(N))
    
    # Icosahedron-based
    initializations.append(_icosahedron_subdivide(N, rng))
    
    # Alternative golden spirals with offsets
    for offset in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        initializations.append(_golden_spiral_v2(N, offset))
    
    # Random starts - more diverse
    for seed_offset in range(12):
        rng_sub = np.random.default_rng(42 + seed_offset * 7)
        initializations.append(_random_sphere(N, rng_sub))
    
    # Perturbed Fibonacci
    for sigma in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.2, 0.3]:
        rng_sub = np.random.default_rng(1000 + int(sigma * 100))
        initializations.append(_perturb_sphere(_fibonacci_sphere(N), rng_sub, sigma))
    
    # Perturbed icosahedron
    for sigma in [0.01, 0.02, 0.05, 0.08, 0.1, 0.15]:
        rng_sub = np.random.default_rng(2000 + int(sigma * 100))
        initializations.append(_perturb_sphere(_icosahedron_subdivide(N, rng_sub), rng_sub, sigma))
    
    # Perturbed golden spirals
    for offset in [0.1, 0.3, 0.5, 0.7]:
        for sigma in [0.02, 0.05, 0.1]:
            rng_sub = np.random.default_rng(3000 + int(offset * 100) + int(sigma * 1000))
            initializations.append(_perturb_sphere(_golden_spiral_v2(N, offset), rng_sub, sigma))
    
    # Additional structured initializations
    # Sphere with z-axis symmetry breaking
    rng_sym = np.random.default_rng(5000)
    pts_sym = _fibonacci_sphere(N)
    # Rotate by random angle
    theta = rng_sym.uniform(0, 2 * np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    pts_rot = _project_to_sphere(pts_sym @ rot.T)
    initializations.append(pts_rot)
    
    # Double perturbation for more diversity
    for sigma1 in [0.05, 0.1]:
        for sigma2 in [0.02, 0.05]:
            rng_sub = np.random.default_rng(6000 + int(sigma1 * 100) + int(sigma2 * 1000))
            pts_pert1 = _perturb_sphere(_fibonacci_sphere(N), rng_sub, sigma1)
            pts_pert2 = _perturb_sphere(pts_pert1, rng_sub, sigma2)
            initializations.append(pts_pert2)
    
    init_energies = [_log_energy(p) for p in initializations]
    sorted_indices = np.argsort(init_energies)
    
    best_pts = initializations[sorted_indices[0]]
    best_energy = init_energies[sorted_indices[0]]
    
    # Process initializations in order of initial energy
    # Focus on top candidates with more time each
    num_to_try = min(len(sorted_indices), 15)
    for rank in range(num_to_try):
        idx = sorted_indices[rank]
        if time.time() > deadline - 20.0:
            break
        
        init_pts = initializations[idx]
        # Allocate time based on rank: top gets most time
        remaining = deadline - time.time()
        if rank == 0:
            time_alloc = min(30.0, remaining * 0.4)
        elif rank < 3:
            time_alloc = min(20.0, remaining * 0.3)
        elif rank < 7:
            time_alloc = min(12.0, remaining * 0.2)
        else:
            time_alloc = min(8.0, remaining * 0.15)
        
        if time_alloc < 3.0:
            break
            
        pts, energy = _optimize_from_start(init_pts, time_alloc, deadline)
        
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Final polish on the best result
    if time.time() < deadline - 8.0:
        try:
            pts_final, e_final = _optimize_from_start(best_pts, 20.0, deadline)
            if e_final < best_energy:
                best_pts = pts_final
                best_energy = e_final
        except Exception:
            pass
    
    # Ultra-final polish
    if time.time() < deadline - 3.0:
        try:
            pts_ultra, e_ultra = _riemannian_descent(best_pts, max_iter=1500, tol=1e-14, deadline=deadline - 3.0)
            if e_ultra < best_energy:
                best_pts = pts_ultra
                best_energy = e_ultra
        except Exception:
            pass
    
    best_pts = _project_to_sphere(best_pts)
    
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    return best_pts

# EVOLVE-BLOCK-END
