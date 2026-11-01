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
    
    grad = pts * sum_inv_d2[:, np.newaxis] - sum_xj_inv_d2
    
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

def _optimize_from_start(pts0, time_budget, deadline):
    """Optimize from a starting configuration."""
    best_pts = _project_to_sphere(pts0.copy())
    best_energy = _log_energy(best_pts)
    
    t0 = time.time()
    
    def objective_and_grad(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e = _log_energy(pts)
        g = _gradient_log_energy(pts)
        norms = np.linalg.norm(x.reshape(N, 3), axis=1)
        norms = np.maximum(norms, 1e-15)
        grad_x = g / norms[:, np.newaxis]
        return e, grad_x.reshape(-1)
    
    x0 = best_pts.reshape(-1)
    
    # First pass: L-BFGS-B with moderate tolerance
    try:
        res = minimize(
            lambda x: objective_and_grad(x)[0],
            x0,
            jac=lambda x: objective_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 300, 'gtol': 1e-10, 'maxfun': 600, 'ftol': 1e-12}
        )
        if res.success or res.fun < best_energy:
            pts_new = _project_to_sphere(res.x.reshape(N, 3))
            e_new = _log_energy(pts_new)
            if e_new < best_energy:
                best_pts = pts_new
                best_energy = e_new
    except Exception:
        pass
    
    if time.time() > deadline - 5.0:
        return best_pts, best_energy
    
    # Second pass: tighter tolerance
    try:
        x1 = best_pts.reshape(-1)
        res2 = minimize(
            lambda x: objective_and_grad(x)[0],
            x1,
            jac=lambda x: objective_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-14, 'maxfun': 1000, 'ftol': 1e-15}
        )
        if res2.fun < best_energy:
            pts_new = _project_to_sphere(res2.x.reshape(N, 3))
            e_new = _log_energy(pts_new)
            if e_new < best_energy:
                best_pts = pts_new
                best_energy = e_new
    except Exception:
        pass
    
    # Third pass: very tight tolerance for final polish
    if time.time() < deadline - 2.0:
        try:
            x2 = best_pts.reshape(-1)
            res3 = minimize(
                lambda x: objective_and_grad(x)[0],
                x2,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 800, 'gtol': 1e-16, 'maxfun': 1500, 'ftol': 1e-16}
            )
            if res3.fun < best_energy:
                pts_new = _project_to_sphere(res3.x.reshape(N, 3))
                e_new = _log_energy(pts_new)
                if e_new < best_energy:
                    best_pts = pts_new
                    best_energy = e_new
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
    for offset in [0.1, 0.3, 0.5]:
        initializations.append(_golden_spiral_v2(N, offset))
    
    # Random starts
    for seed_offset in range(8):
        rng_sub = np.random.default_rng(42 + seed_offset)
        initializations.append(_random_sphere(N, rng_sub))
    
    # Perturbed Fibonacci
    for sigma in [0.01, 0.03, 0.08, 0.15, 0.25]:
        rng_sub = np.random.default_rng(1000 + int(sigma * 100))
        initializations.append(_perturb_sphere(_fibonacci_sphere(N), rng_sub, sigma))
    
    # Perturbed icosahedron
    for sigma in [0.02, 0.05, 0.1]:
        rng_sub = np.random.default_rng(2000 + int(sigma * 100))
        initializations.append(_perturb_sphere(_icosahedron_subdivide(N, rng_sub), rng_sub, sigma))
    
    # Perturbed golden spirals
    for offset in [0.1, 0.3]:
        for sigma in [0.02, 0.05]:
            rng_sub = np.random.default_rng(3000 + int(offset * 100) + int(sigma * 1000))
            initializations.append(_perturb_sphere(_golden_spiral_v2(N, offset), rng_sub, sigma))
    
    best_pts = initializations[0]
    best_energy = _log_energy(best_pts)
    
    init_energies = [_log_energy(p) for p in initializations]
    sorted_indices = np.argsort(init_energies)
    
    for idx in sorted_indices:
        if time.time() > deadline - 10.0:
            break
        
        init_pts = initializations[idx]
        pts, energy = _optimize_from_start(init_pts, 20.0, deadline)
        
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Final polish on the best result
    if time.time() < deadline - 3.0:
        try:
            pts_final, e_final = _optimize_from_start(best_pts, 10.0, deadline)
            if e_final < best_energy:
                best_pts = pts_final
                best_energy = e_final
        except Exception:
            pass
    
    best_pts = _project_to_sphere(best_pts)
    
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    return best_pts

# EVOLVE-BLOCK-END
