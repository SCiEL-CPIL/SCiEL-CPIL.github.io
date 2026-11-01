"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

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

def _energy_and_tangent_grad(pts):
    """Compute energy and tangent-space gradient of logarithmic energy.
    
    E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    
    The R^3 gradient is: dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    (negative because increasing distance decreases energy)
    
    Tangent gradient: project onto tangent space.
    """
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    
    # Energy
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    log_d = np.log(dist[iu])
    energy = -math.fsum(log_d)
    
    # Gradient in R^3: dE/dx_i = -sum_{j!=i} (x_i - x_j) / d2_ij
    # = -[x_i * sum_{j!=i} 1/d2_ij - sum_{j!=i} x_j / d2_ij]
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    sum_inv_d2 = inv_d2.sum(axis=1)  # (n,)
    sum_xj_inv_d2 = (pts.T @ inv_d2).T  # (n, 3)
    
    # R^3 gradient (with negative sign)
    grad_r3 = -(pts * sum_inv_d2[:, np.newaxis] - sum_xj_inv_d2)
    
    # Project onto tangent space
    grad_dot_xi = np.sum(grad_r3 * pts, axis=1, keepdims=True)
    grad_tan = grad_r3 - grad_dot_xi * pts
    
    return energy, grad_tan

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _retract(pts):
    """Retraction to sphere: normalize."""
    return _project_to_sphere(pts)

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

def _icosahedron_subdivide(n_target, rng):
    """Generate points via icosahedron subdivision for better initial distribution."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Subdivide until we have enough points
    current = verts.copy()
    while len(current) < n_target:
        new_points = []
        # For each edge, add midpoint if not already present
        # Simple approach: subdivide all faces
        # Use a simpler method: just add random points if we're close
        if len(current) > 50:
            # Add remaining points randomly near existing structure
            remaining = n_target - len(current)
            for _ in range(remaining):
                idx = rng.integers(len(current))
                base = current[idx]
                noise = rng.standard_normal(3) * 0.1
                new_pt = _project_to_sphere(base + noise)
                new_points.append(new_pt)
            current = np.vstack([current, np.array(new_points)])
            break
        else:
            # Subdivide by adding midpoints of edges
            midpoints = []
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    d = np.linalg.norm(current[i] - current[j])
                    if d < 0.5:  # Only add if edge is short enough
                        mid = _project_to_sphere(current[i] + current[j])
                        # Check if midpoint already exists
                        is_dup = False
                        for k in range(len(current)):
                            if np.linalg.norm(mid - current[k]) < 0.01:
                                is_dup = True
                                break
                        if not is_dup:
                            midpoints.append(mid)
            if len(midpoints) == 0:
                # Fallback: add random points
                remaining = n_target - len(current)
                for _ in range(remaining):
                    new_points.append(_random_sphere(1, rng)[0])
                current = np.vstack([current, np.array(new_points)])
                break
            current = np.vstack([current, np.array(midpoints)])
    
    # If we have too many, randomly select
    if len(current) > n_target:
        indices = rng.choice(len(current), size=n_target, replace=False)
        current = current[indices]
    
    return current[:n_target]

def _riemannian_gradient_descent(pts, max_iter=500, tol=1e-12, time_limit=None, t_start=None):
    """Riemannian gradient descent with adaptive step size."""
    best_pts = _project_to_sphere(pts.copy())
    best_energy = _log_energy(best_pts)
    
    if t_start is None:
        t_start = time.time()
    
    step_size = 0.01
    prev_energy = best_energy
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - t_start > time_limit:
            break
        
        energy, grad_tan = _energy_and_tangent_grad(best_pts)
        
        if energy < best_energy:
            best_energy = energy
        
        # Adaptive step size
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < tol:
            break
        
        # Try the step
        new_pts = _project_to_sphere(best_pts - step_size * grad_tan)
        new_energy = _log_energy(new_pts)
        
        if new_energy < energy:
            # Accept step, maybe increase step size
            best_pts = new_pts
            step_size = min(step_size * 1.1, 0.1)
        else:
            # Reject step, decrease step size
            step_size *= 0.5
            if step_size < 1e-10:
                break
        
        # Convergence check
        if abs(prev_energy - energy) < tol * max(1.0, abs(energy)):
            if it > 10:
                break
        prev_energy = energy
    
    return best_pts, best_energy

def _optimize_from_start(pts0, time_budget):
    """Optimize from a starting configuration using multiple methods."""
    best_pts = _project_to_sphere(pts0.copy())
    best_energy = _log_energy(best_pts)
    
    t0 = time.time()
    
    # Method 1: L-BFGS-B with corrected gradient
    def objective_and_grad(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, g_tan = _energy_and_tangent_grad(pts)
        # Chain rule: grad_x = g_tan / |x_i|
        norms = np.linalg.norm(x.reshape(N, 3), axis=1)
        norms = np.maximum(norms, 1e-15)
        grad_x = g_tan / norms[:, np.newaxis]
        return e, grad_x.reshape(-1)
    
    x0 = best_pts.reshape(-1)
    
    try:
        res = minimize(
            lambda x: objective_and_grad(x)[0],
            x0,
            jac=lambda x: objective_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 300, 'gtol': 1e-10, 'maxfun': 500, 'ftol': 1e-12}
        )
        pts_new = _project_to_sphere(res.x.reshape(N, 3))
        e_new = _log_energy(pts_new)
        if e_new < best_energy:
            best_pts = pts_new
            best_energy = e_new
    except Exception:
        pass
    
    # Method 2: Riemannian gradient descent polish
    remaining = time_budget - (time.time() - t0)
    if remaining > 2.0:
        try:
            pts_rg, e_rg = _riemannian_gradient_descent(
                best_pts, max_iter=300, tol=1e-12,
                time_limit=min(remaining * 0.5, 10.0), t_start=time.time()
            )
            if e_rg < best_energy:
                best_pts = pts_rg
                best_energy = e_rg
        except Exception:
            pass
    
    # Method 3: Final high-precision L-BFGS polish
    remaining = time_budget - (time.time() - t0)
    if remaining > 2.0:
        try:
            x1 = best_pts.reshape(-1)
            res2 = minimize(
                lambda x: objective_and_grad(x)[0],
                x1,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-14, 'maxfun': 800, 'ftol': 1e-15}
            )
            pts_new = _project_to_sphere(res2.x.reshape(N, 3))
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
    
    # Fibonacci spiral
    initializations.append(_fibonacci_sphere(N))
    
    # Icosahedron subdivision
    initializations.append(_icosahedron_subdivide(N, rng))
    
    # Random starts
    for _ in range(4):
        initializations.append(_random_sphere(N, rng))
    
    # Perturbed Fibonacci
    for sigma in [0.03, 0.08, 0.15]:
        initializations.append(_perturb_sphere(_fibonacci_sphere(N), rng, sigma))
    
    # Perturbed icosahedron
    for sigma in [0.03, 0.08]:
        initializations.append(_perturb_sphere(_icosahedron_subdivide(N, rng), rng, sigma))
    
    # More random
    for _ in range(3):
        initializations.append(_random_sphere(N, rng))
    
    best_pts = initializations[0]
    best_energy = _log_energy(best_pts)
    
    # Try each initialization
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 15.0:
            break
        
        remaining = deadline - time.time()
        # Allocate time based on remaining, cap at 20s per init
        time_for_this = min(remaining * 0.7, 20.0)
        
        if time_for_this < 3.0:
            continue
        
        pts, energy = _optimize_from_start(init_pts, time_for_this)
        
        if energy < best_energy:
            best_pts = pts
            best_energy = energy
    
    # Final polish on the best result
    if time.time() < deadline - 5.0:
        try:
            pts_final, e_final = _optimize_from_start(best_pts, deadline - time.time() - 2.0)
            if e_final < best_energy:
                best_pts = pts_final
                best_energy = e_final
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Validate
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    # Check for duplicate points
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            min_dist = min(min_dist, d)
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci if points are too close
        return _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
