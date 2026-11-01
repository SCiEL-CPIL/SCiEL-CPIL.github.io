"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(points)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return 1e10
            total += math.log(1.0 / d)
    return total

def _log_energy_and_grad(points):
    """Compute logarithmic energy and Euclidean gradient."""
    n = len(points)
    energy = 0.0
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i + 1, n):
            diff = points[i] - points[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                return 1e10, grad
            # d/dx_i log(1/d) = d/dx_i (-log(d)) = -(diff_i / d) / d = -diff_i / d^2
            # Wait: log(1/d) = -log(d), d/dx_i (-log(d)) where d = |x_i - x_j|
            # = -1/d * d(d)/dx_i = -1/d * (x_i - x_j)/d = -(x_i - x_j)/d^2
            factor = 1.0 / (d * d)
            energy += -math.log(d)
            grad[i] += -factor * diff
            grad[j] += factor * diff
    return energy, grad

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms

def _riemannian_gradient(points, euclidean_grad):
    """Project Euclidean gradient to tangent space."""
    # Tangent space: remove radial component
    radial = np.sum(euclidean_grad * points, axis=1, keepdims=True)
    return euclidean_grad - radial * points

def _riemannian_step(points, grad, step_size):
    """Take a Riemannian gradient descent step."""
    # Update in Euclidean space, then project back
    new_points = points - step_size * grad
    return _project_to_sphere(new_points)

def _optimize_from_start(points, max_iter=5000, tol=1e-12):
    """Optimize points on sphere using Riemannian gradient descent with adaptive step size."""
    points = _project_to_sphere(points.copy())
    energy, grad = _log_energy_and_grad(points)
    rgrad = _riemannian_gradient(points, grad)
    
    step_size = 0.01
    best_energy = energy
    best_points = points.copy()
    
    for it in range(max_iter):
        # Check convergence
        gnorm = np.linalg.norm(rgrad)
        if gnorm < tol:
            break
        
        # Adaptive step size with backtracking
        alpha = step_size
        for _ in range(20):
            new_points = _riemannian_step(points, rgrad, alpha)
            new_energy, new_grad = _log_energy_and_grad(new_points)
            if new_energy < energy:
                break
            alpha *= 0.5
        
        if new_energy < energy:
            points = new_points
            energy = new_energy
            grad = new_grad
            rgrad = _riemannian_gradient(points, grad)
            step_size = min(alpha * 1.5, 0.1)
            
            if energy < best_energy:
                best_energy = energy
                best_points = points.copy()
        else:
            step_size *= 0.5
            if step_size < 1e-10:
                break
    
    return best_points, best_energy

def _polish_with_lbfgsb(points, max_iter=1000):
    """Polish using L-BFGS-B with penalty for sphere constraint."""
    n = len(points)
    
    def objective(flat):
        pts = _project_to_sphere(flat.reshape(n, 3))
        energy, _ = _log_energy_and_grad(pts)
        # Add small penalty for deviation from unit sphere (should be ~0 after projection)
        return energy
    
    def gradient(flat):
        pts = _project_to_sphere(flat.reshape(n, 3))
        _, grad = _log_energy_and_grad(pts)
        # Chain rule through projection: d(project)/d(flat)
        # project(x) = x/|x|, so d(project)/d(x) = (I - xx^T)/|x|^3 ... 
        # Actually for flat input, we need to be more careful
        # Let's just use the Riemannian gradient projected back
        rgrad = _riemannian_gradient(pts, grad)
        # The gradient with respect to flat coordinates
        # This is approximate but works in practice
        return rgrad.flatten()
    
    result = minimize(objective, points.flatten(), jac=gradient, 
                      method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})
    
    if result.success or result.fun < _log_energy(points):
        polished = _project_to_sphere(result.x.reshape(n, 3))
        return polished
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_points = None
    best_energy = float('inf')
    
    # Generate multiple diverse initializations
    initializations = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(fib_points)
    
    # 2. Random points
    np.random.seed(42)
    for _ in range(5):
        rand_pts = np.random.randn(N, 3)
        rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1, keepdims=True)
        initializations.append(rand_pts)
    
    # 3. Icosahedral-based: start with icosahedron vertices and subdivide
    # Simple approach: use spherical Fibonacci with different parameters
    for seed_val in [1.0, 2.0, 3.0]:
        i = np.arange(N) + seed_val
        z = 1.0 - 2.0 * i / (N + 2*seed_val)
        z = np.clip(z, -1, 1)
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        initializations.append(pts)
    
    # 4. Concentric rings
    n_rings = 10
    pts_per_ring = N // n_rings
    ring_pts = []
    for k in range(n_rings):
        z_val = 1.0 - 2.0 * (k + 0.5) / n_rings
        r_val = np.sqrt(1.0 - z_val * z_val)
        for m in range(pts_per_ring):
            angle = 2.0 * np.pi * m / pts_per_ring + np.pi * k / n_rings
            ring_pts.append([r_val * np.cos(angle), r_val * np.sin(angle), z_val])
    ring_pts = np.array(ring_pts[:N])
    initializations.append(ring_pts)
    
    # Optimize each initialization
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 10.0:
            break
        
        # Riemannian optimization
        opt_pts, opt_energy = _optimize_from_start(init_pts, max_iter=3000)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_points = opt_pts.copy()
        
        # Polish with L-BFGS-B if time permits
        if time.time() > deadline - 5.0:
            break
        
        polished = _polish_with_lbfgsb(opt_pts, max_iter=500)
        pol_energy = _log_energy(polished)
        
        if pol_energy < best_energy:
            best_energy = pol_energy
            best_points = polished.copy()
    
    # Final safety check
    if best_points is None:
        best_points = initializations[0]
    
    # Ensure all points are on unit sphere and valid
    best_points = _project_to_sphere(best_points)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Nudge apart
                best_points[j] += 1e-8 * (best_points[j] - best_points[i]) / (d + 1e-15)
                best_points[j] = _project_to_sphere(best_points[j].reshape(1, 3))[0]
    
    return best_points
# EVOLVE-BLOCK-END

