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
    """Tangent-space gradient of logarithmic energy.
    
    E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    
    The R^3 gradient is: dE/dx_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    Tangent gradient: project onto tangent space.
    """
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    sum_inv_d2 = inv_d2.sum(axis=1)
    sum_xj_inv_d2 = (pts.T @ inv_d2).T
    
    # R^3 gradient: sum_{j!=i} (x_j - x_i) / d2_ij
    grad_r3 = sum_xj_inv_d2 - pts * sum_inv_d2[:, np.newaxis]
    
    # Project onto tangent space
    grad_dot_xi = np.sum(grad_r3 * pts, axis=1, keepdims=True)
    grad_tan = grad_r3 - grad_dot_xi * pts
    
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
    
    pts = list(verts)
    
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

def _riemannian_gradient_descent(pts, max_iter=500, tol=1e-12, deadline=None):
    """Riemannian gradient descent with adaptive step size."""
    best_pts = _project_to_sphere(pts.copy())
    best_energy = _log_energy(best_pts)
    
    step_size = 0.01
    prev_energy = best_energy
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad_tan = _log_energy(best_pts), _gradient_log_energy(best_pts)
        
        if energy < best_energy:
            best_energy = energy
        
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < tol:
            break
        
        new_pts = _project_to_sphere(best_pts - step_size * grad_tan)
        new_energy = _log_energy(new_pts)
        
        if new_energy < energy:
            best_pts = new_pts
            step_size = min(step_size * 1.1, 0.1)
        else:
            step_size *= 0.5
            if step_size < 1e-10:
                break
        
        if abs(prev_energy - energy) < tol * max(1.0, abs(energy)):
            if it > 10:
                break
        prev_energy = energy
    
    return best_pts, best_energy

def _riemannian_conjugate_gradient(pts, max_iter=300, tol=1e-13, deadline=None):
    """Riemannian conjugate gradient method for faster convergence."""
    X = _project_to_sphere(pts.copy())
    best_pts = X.copy()
    best_energy = _log_energy(X)
    
    energy, g = _log_energy(X), _gradient_log_energy(X)
    d = -g
    d_norm = np.linalg.norm(d)
    if d_norm < tol:
        return X, best_energy
    
    prev_g = g.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        alpha = 0.05
        for ls in range(30):
            new_pts = _project_to_sphere(X - alpha * d)
            new_energy = _log_energy(new_pts)
            if new_energy < energy - 1e-4 * alpha * np.sum(g * d):
                break
            alpha *= 0.5
        else:
            break
        
        X = new_pts
        energy = new_energy
        
        if energy < best_energy:
            best_energy = energy
            best_pts = X.copy()
        
        new_g = _gradient_log_energy(X)
        
        if np.linalg.norm(new_g) < tol:
            break
        
        beta = np.sum(new_g * new_g) / max(np.sum(prev_g * prev_g), 1e-30)
        beta = max(beta, 0.0)
        
        d = -new_g + beta * d
        
        if np.sum(d * new_g) > 0:
            d = -new_g
        
        prev_g = new_g
    
    return best_pts, best_energy

def _optimize_from_start(pts0, time_budget, deadline):
    """Optimize from a starting configuration using multiple methods."""
    best_pts = _project_to_sphere(pts0.copy())
    best_energy = _log_energy(best_pts)
    
    # Method 1: L-BFGS-B with tangent gradient (projection handles constraint)
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        return _log_energy(pts)
    
    def gradient(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        g = _gradient_log_energy(pts)
        return g.reshape(-1)
    
    x0 = best_pts.reshape(-1)
    
    try:
        res = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-8, 'maxfun': 800, 'ftol': 1e-10}
        )
        pts_new = _project_to_sphere(res.x.reshape(N, 3))
        e_new = _log_energy(pts_new)
        if e_new < best_energy:
            best_pts = pts_new
            best_energy = e_new
    except Exception:
        pass
    
    # Method 2: Riemannian conjugate gradient polish
    remaining = deadline - time.time()
    if remaining > 5.0:
        try:
            pts_rcg, e_rcg = _riemannian_conjugate_gradient(
                best_pts, max_iter=200, tol=1e-13,
                deadline=min(deadline - 5.0, time.time() + remaining * 0.4)
            )
            if e_rcg < best_energy:
                best_pts = pts_rcg
                best_energy = e_rcg
        except Exception:
            pass
    
    # Method 3: Riemannian gradient descent fine-tune
    remaining = deadline - time.time()
    if remaining > 3.0:
        try:
            pts_rg, e_rg = _riemannian_gradient_descent(
                best_pts, max_iter=200, tol=1e-13,
                deadline=min(deadline - 3.0, time.time() + remaining * 0.3)
            )
            if e_rg < best_energy:
                best_pts = pts_rg
                best_energy = e_rg
        except Exception:
            pass
    
    # Method 4: Final high-precision L-BFGS polish
    remaining = deadline - time.time()
    if remaining > 3.0:
        try:
            x1 = best_pts.reshape(-1)
            res2 = minimize(
                objective,
                x1,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 800, 'gtol': 1e-12, 'maxfun': 1200, 'ftol': 1e-14}
            )
            pts_new = _project_to_sphere(res2.reshape(N, 3))
            e_new = _log_energy(pts_new)
            if e_new < best_energy:
                best_pts = pts_new
                best_energy = e_new
        except Exception:
            pass
    
    return best_pts, best_energy

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spiral (good baseline)
    initializations.append(_fibonacci_sphere(N))
    
    # Icosahedron subdivision
    initializations.append(_icosahedron_subdivide(N, rng))
    
    # Perturbed Fibonacci with different scales
    for sigma in [0.05, 0.12]:
        initializations.append(_perturb_sphere(_fibonacci_sphere(N), rng, sigma))
    
    # Perturbed icosahedron
    initializations.append(_perturb_sphere(_icosahedron_subdivide(N, rng), rng, 0.05))
    
    # A few random starts for diversity
    for _ in range(3):
        initializations.append(_random_sphere(N, rng))
    
    # Sort initializations by initial energy (best first)
    init_energies = [(init_pts, _log_energy(init_pts)) for init_pts in initializations]
    init_energies.sort(key=lambda x: x[1])
    
    best_pts = init_energies[0][0]
    best_energy = init_energies[0][1]
    
    # Try each initialization with adaptive time allocation
    n_inits = len(init_energies)
    for idx, (init_pts, init_e) in enumerate(init_energies):
        if time.time() > deadline - 20.0:
            break
        
        remaining = deadline - time.time()
        if idx == 0:
            time_for_this = min(remaining * 0.4, 30.0)
        elif idx < 3:
            time_for_this = min(remaining * 0.25, 20.0)
        else:
            time_for_this = min(remaining * 0.15, 12.0)
        
        if time_for_this < 3.0:
            continue
        
        pts, energy = _optimize_from_start(init_pts, time_for_this, deadline)
        
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Final polish on the best result
    if time.time() < deadline - 5.0:
        try:
            pts_final, e_final = _optimize_from_start(best_pts, deadline - time.time() - 2.0, deadline)
            if e_final < best_energy:
                best_pts = pts_final
                best_energy = e_final
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    # Check for duplicate points
    dots = best_pts @ best_pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    min_dist = np.sqrt(np.min(d2))
    
    if min_dist < 1e-9:
        return _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
