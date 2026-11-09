"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _log_energy(pts):
    """Compute log(1/d) energy sum over pairs, with numerical safety."""
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Zero out diagonal to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # Clip to avoid log of very small numbers causing issues
    dists = np.maximum(dists, 1e-15)
    # Sum over upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    vals = -np.log(dists[iu])
    return float(math.fsum(vals))

def _log_energy_grad(pts):
    """Compute gradient of log energy w.r.t. point positions (not projected)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    dists = np.maximum(dists, 1e-15)
    
    # Gradient of -log(d_ij) w.r.t. x_i is diff_ij / d_ij^2
    # For each pair (i,j), contribution to grad[i] is (x_i - x_j) / d_ij^2
    # contribution to grad[j] is (x_j - x_i) / d_ij^2 = -diff_ij / d_ij^2
    
    # diff[i,j] = x_i - x_j
    # grad[i] += sum_j (x_i - x_j) / d_ij^2
    # grad[j] -= sum_i (x_i - x_j) / d_ij^2
    
    inv_d2 = 1.0 / (dists ** 2)
    inv_d2[np.diag_indices(n)] = 0.0
    
    # grad[i] = sum_j diff[i,j] * inv_d2[i,j]
    grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    return grad

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _spherical_energy_and_grad(theta_phi):
    """
    theta_phi: (n, 2) array of (theta, phi) angles
    Returns (energy, gradient in (theta, phi) space)
    """
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]
    
    # Convert to Cartesian
    sin_t = np.sin(theta)
    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = np.cos(theta)
    pts = np.column_stack([x, y, z])
    
    energy = _log_energy(pts)
    grad_cart = _log_energy_grad(pts)
    
    # Chain rule: grad_theta, grad_phi
    # dx/dtheta = cos(theta)cos(phi), dy/dtheta = cos(theta)sin(phi), dz/dtheta = -sin(theta)
    # dx/dphi = -sin(theta)sin(phi), dy/dphi = sin(theta)cos(phi), dz/dphi = 0
    
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    
    grad_theta = grad_cart[:, 0] * ct * cp + grad_cart[:, 1] * ct * sp + grad_cart[:, 2] * (-st)
    grad_phi = grad_cart[:, 0] * (-st * sp) + grad_cart[:, 1] * (st * cp) + grad_cart[:, 2] * 0.0
    
    grad = np.column_stack([grad_theta, grad_phi])
    return energy, grad

def _cartesian_energy_and_grad(pts):
    """
    pts: (n, 3) array of points in Cartesian space
    Returns (energy, gradient in Cartesian space)
    """
    energy = _log_energy(pts)
    grad = _log_energy_grad(pts)
    return energy, grad

def _fibonacci_spiral():
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere_points(rng, n):
    """Generate n random points on unit sphere."""
    theta = rng.uniform(0, 2 * np.pi, size=n)
    z = rng.uniform(-1, 1, size=n)
    r = np.sqrt(1 - z**2)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])

def _cartesian_gradient_descent(pts, max_iter=500, time_limit=20.0, verbose=False):
    """
    Perform gradient descent in Cartesian space with projection to sphere.
    Uses adaptive step size.
    """
    start_time = time.time()
    pts = pts.copy()
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    
    # Initial step size
    step = 0.01
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _cartesian_energy_and_grad(pts)
        
        # Project gradient to tangent space: grad_tangent = grad - (grad . x) * x
        dot = np.sum(pts * grad, axis=1, keepdims=True)
        grad_tangent = grad - dot * pts
        
        # Adaptive step size based on gradient magnitude
        grad_norm = np.linalg.norm(grad_tangent, axis=1)
        max_grad = np.max(grad_norm)
        if max_grad > 1e-10:
            step = min(0.05, 0.01 / (max_grad * 0.1 + 1))
        
        # Update
        pts_new = pts - step * grad_tangent
        pts_new = _project_to_sphere(pts_new)
        
        energy_new = _log_energy(pts_new)
        
        # Accept if improved
        if energy_new < energy:
            pts = pts_new
            if energy_new < best_energy:
                best_energy = energy_new
                best_pts = pts_new.copy()
        else:
            # Reduce step size
            step *= 0.5
        
        if verbose and it % 100 == 0:
            print(f"Iter {it}: energy={energy:.10f}, best={best_energy:.10f}")
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (10s before 150s limit)
    
    best_energy = float('inf')
    best_pts = None
    
    # Strategy 1: Fibonacci spiral + L-BFGS-B
    if time.time() < deadline - 60.0:
        pts = _fibonacci_spiral()
        theta = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))
        phi_s = np.arctan2(pts[:, 1], pts[:, 0])
        tp = np.column_stack([theta, phi_s])
        
        from scipy.optimize import minimize
        
        def objective(tp_flat):
            tp_2d = tp_flat.reshape(-1, 2)
            return _spherical_energy_and_grad(tp_2d)[0]
        
        def gradient(tp_flat):
            tp_2d = tp_flat.reshape(-1, 2)
            return _spherical_energy_and_grad(tp_2d)[1].ravel()
        
        result = minimize(
            objective,
            tp.ravel(),
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        theta = result.x.reshape(-1, 2)[:, 0]
        phi = result.x.reshape(-1, 2)[:, 1]
        sin_t = np.sin(theta)
        x = sin_t * np.cos(phi)
        y = sin_t * np.sin(phi)
        z = np.cos(theta)
        pts = np.column_stack([x, y, z])
        pts = _project_to_sphere(pts)
        
        energy = _log_energy(pts)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Strategy 2: Cartesian gradient descent from Fibonacci start
    if time.time() < deadline - 30.0:
        pts = _fibonacci_spiral()
        pts_gd, energy_gd = _cartesian_gradient_descent(pts, max_iter=1000, time_limit=30.0)
        if energy_gd < best_energy:
            best_energy = energy_gd
            best_pts = pts_gd.copy()
    
    # Strategy 3: Multiple random restarts with gradient descent
    rng = np.random.default_rng(42)
    for restart in range(10):
        if time.time() > deadline - 15.0:
            break
        
        pts = _random_sphere_points(rng, N)
        pts_gd, energy_gd = _cartesian_gradient_descent(pts, max_iter=500, time_limit=10.0)
        if energy_gd < best_energy:
            best_energy = energy_gd
            best_pts = pts_gd.copy()
    
    # Final polishing: small step gradient descent on best solution
    if time.time() < deadline - 5.0 and best_pts is not None:
        pts_polish, energy_polish = _cartesian_gradient_descent(
            best_pts, max_iter=2000, time_limit=10.0
        )
        if energy_polish < best_energy:
            best_energy = energy_polish
            best_pts = pts_polish.copy()
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_spiral()
    
    # Final normalization and validation
    best_pts = _project_to_sphere(best_pts)
    
    # Check for any points too close together
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge apart
                best_pts[j] += 1e-8 * rng.normal(size=3)
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
# EVOLVE-BLOCK-END
