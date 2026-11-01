"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|xi-xj|)."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient on the sphere.
    Energy: E = sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log|xi - xj|
    Gradient on the sphere (projected):
    For point i, grad_i = sum_{j != i} (xi - xj) / |xi - xj|^2
    (This is the gradient of -sum_j log|xi-xj| with respect to xi, projected to tangent space)
    
    Actually, let's be careful:
    E = sum_{i<j} -log|xi - xj|
    dE/dxi = sum_{j != i} (xi - xj) / |xi - xj|^2
    
    We need to project this to the tangent space at xi:
    grad_proj_i = grad_i - (grad_i . xi) * xi
    """
    n = len(pts)
    energy = 0.0
    grad = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-15:
                return float('inf'), grad
            # Contribution to energy
            energy -= math.log(d)
            # Gradient contributions
            # dE/dxi += (xi - xj) / d^2
            # dE/dxj += (xj - xi) / d^2 = -(xi - xj) / d^2
            inv_d2 = 1.0 / (d * d)
            grad[i] += diff * inv_d2
            grad[j] -= diff * inv_d2
    
    # Project gradients to tangent space
    for i in range(n):
        grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
    
    return energy, grad

def _normalize(pts):
    """Normalize each point to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize(pts, time_limit, tol=1e-14):
    """
    Optimize points on the sphere using projected gradient descent with adaptive step size.
    """
    start_time = time.time()
    n = len(pts)
    
    # Initial energy
    energy, grad = _energy_and_grad(pts)
    best_energy = energy
    best_pts = pts.copy()
    
    # Adaptive step size
    step_size = 0.01
    min_step = 1e-16
    max_step = 0.5
    
    while time.time() - start_time < time_limit - 5.0:
        # Project gradient descent step
        new_pts = pts - step_size * grad
        new_pts = _normalize(new_pts)
        
        new_energy, new_grad = _energy_and_grad(new_pts)
        
        if not math.isfinite(new_energy):
            step_size *= 0.5
            if step_size < min_step:
                break
            continue
        
        if new_energy < energy:
            # Accept step
            pts = new_pts
            grad = new_grad
            energy = new_energy
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
            
            # Increase step size slightly
            step_size = min(step_size * 1.05, max_step)
        else:
            # Reject step, decrease step size
            step_size *= 0.5
            if step_size < min_step:
                break
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
    
    return best_pts, best_energy

def construct():
    """
    Construct 150 points on the unit sphere to minimize logarithmic energy.
    """
    start_time = time.time()
    total_time_limit = 140.0  # Leave 10s margin
    
    # Generate initial points
    pts = _fibonacci_sphere(N)
    
    # Phase 1: Coarse optimization
    phase1_time = min(60.0, total_time_limit * 0.5)
    pts, energy = _optimize(pts, phase1_time, tol=1e-12)
    
    # Phase 2: Fine optimization
    remaining_time = total_time_limit - (time.time() - start_time)
    if remaining_time > 10.0:
        pts, energy2 = _optimize(pts, remaining_time - 5.0, tol=1e-14)
        if energy2 < energy:
            pts, energy = pts, energy2
    
    # Final normalization
    pts = _normalize(pts)
    
    return pts
# EVOLVE-BLOCK-END
