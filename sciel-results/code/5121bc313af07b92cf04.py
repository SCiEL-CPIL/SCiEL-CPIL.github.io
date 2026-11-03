"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _cartesian_from_spherical(theta, phi):
    """Convert spherical coordinates (theta, phi) to Cartesian coordinates on the unit sphere."""
    # theta: azimuth [0, 2*pi), phi: polar [0, pi]
    sin_phi = np.sin(phi)
    x = sin_phi * np.cos(theta)
    y = sin_phi * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def _log_energy_and_grad_cartesian(points):
    """
    Compute logarithmic energy and its gradient with respect to Cartesian coordinates.
    Energy: sum_{i<j} log(1/|x_i - x_j|)
    Gradient: dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    # diff[i, j] = x_i - x_j
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0) # Avoid division by zero
    
    # d = sqrt(d2)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 1.0)
    
    # Energy terms: log(1/d) = -log(d)
    # We only need i < j
    # For gradient, we need sum over all j != i
    
    # Energy calculation (sum i<j)
    # Use fsum for precision if possible, but for speed in gradient steps, numpy sum is often okay.
    # However, the task specifies math.fsum for the score. We should ensure our internal energy is consistent.
    # For optimization, slight numerical differences are fine as long as the gradient is correct.
    
    # Gradient calculation
    # grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # factor[i, j] = 1 / d2[i, j]
    factor = 1.0 / d2
    np.fill_diagonal(factor, 0.0)
    
    grad_cart = np.sum(diff * factor[:, :, np.newaxis], axis=1)
    
    # Energy calculation
    # We can compute the full sum and divide by 2, or sum upper triangle.
    # Let's sum upper triangle for efficiency and to match the "pair" definition.
    energy = 0.0
    # Using a loop for fsum might be slow, but N=150 -> ~11k pairs. It's fast enough.
    # However, for the optimizer, we need a fast energy function.
    # Let's use numpy sum for the optimizer's internal energy, and fsum only for the final check or if we want to be very precise.
    # Actually, the optimizer just needs a consistent function.
    # Let's use a vectorized sum for the optimizer's objective.
    # E = 0.5 * sum_{i != j} log(1/d_{ij})
    # log(1/d) = -log(d)
    # We can use the matrix d.
    # Mask out diagonal
    mask = ~np.eye(n, dtype=bool)
    # d is symmetric
    # Sum of -log(d) for i != j
    # We can sum the whole matrix and divide by 2.
    # But log(1) = 0 on diagonal, so summing all is fine.
    # E = -0.5 * sum_{i,j} log(d_{ij})
    
    # To avoid log(0) or issues, we already set diagonal d=1.
    # But wait, if points are very close, d is small, log(1/d) is large positive.
    # log(d) is large negative.
    
    # Let's stick to the pair sum for clarity and correctness with fsum in the final step.
    # For the optimizer, we can use a fast approximation.
    # But let's just use the fast numpy sum for the objective function passed to minimize.
    # The final result will be checked with the strict fsum function.
    
    # Fast energy:
    # d is the distance matrix.
    # We need sum_{i<j} -log(d_{ij})
    # triu_indices
    iu, ju = np.triu_indices(n, k=1)
    d_pairs = d[iu, ju]
    # Ensure no zeros
    d_pairs = np.maximum(d_pairs, 1e-15)
    energy = np.sum(-np.log(d_pairs))
    
    return energy, grad_cart

def _spherical_objective(theta_phi_flat):
    """
    Objective function for L-BFGS-B.
    Input: flattened array of [theta_1, phi_1, theta_2, phi_2, ...]
    Output: energy
    """
    theta = theta_phi_flat[0::2]
    phi = theta_phi_flat[1::2]
    
    # Clip phi to [0, pi] to avoid numerical issues
    phi = np.clip(phi, 1e-9, np.pi - 1e-9)
    
    points = _cartesian_from_spherical(theta, phi)
    energy, _ = _log_energy_and_grad_cartesian(points)
    return energy

def _spherical_gradient(theta_phi_flat):
    """
    Gradient of the objective function with respect to theta and phi.
    """
    theta = theta_phi_flat[0::2]
    phi = theta_phi_flat[1::2]
    
    # Clip phi for consistency with objective
    phi = np.clip(phi, 1e-9, np.pi - 1e-9)
    
    points = _cartesian_from_spherical(theta, phi)
    energy, grad_cart = _log_energy_and_grad_cartesian(points)
    
    # Transform gradient from Cartesian (x,y,z) to Spherical (theta, phi)
    # x = sin(phi)cos(theta)
    # y = sin(phi)sin(theta)
    # z = cos(phi)
    
    # dx/dtheta = -sin(phi)sin(theta)
    # dx/dphi = cos(phi)cos(theta)
    # dy/dtheta = sin(phi)cos(theta)
    # dy/dphi = cos(phi)sin(theta)
    # dz/dtheta = 0
    # dz/dphi = -sin(phi)
    
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    gx = grad_cart[:, 0]
    gy = grad_cart[:, 1]
    gz = grad_cart[:, 2]
    
    # dE/dtheta = gx * (-sin_phi sin_theta) + gy * (sin_phi cos_theta)
    dtheta = -gx * sin_phi * sin_theta + gy * sin_phi * cos_theta
    
    # dE/dphi = gx * (cos_phi cos_theta) + gy * (cos_phi sin_theta) + gz * (-sin_phi)
    dphi = gx * cos_phi * cos_theta + gy * cos_phi * sin_theta - gz * sin_phi
    
    # Interleave dtheta and dphi
    grad_flat = np.empty_like(theta_phi_flat)
    grad_flat[0::2] = dtheta
    grad_flat[1::2] = dphi
    
    return grad_flat

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_energy = float('inf')
    best_theta_phi = None
    
    np.random.seed(42)
    
    num_restarts = 10
    
    # Initial guess: Fibonacci sphere in Cartesian, then convert to Spherical
    # Or just random spherical coordinates
    # Let's use a mix: one Fibonacci, rest random
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart == 0:
            # Fibonacci sphere
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi_fib = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
            x = r * np.cos(phi_fib)
            y = r * np.sin(phi_fib)
            
            # Convert to spherical
            theta = np.arctan2(y, x)
            phi = np.arccos(z)
            
            # Ensure theta in [0, 2pi]
            theta[theta < 0] += 2 * np.pi
            
            theta_phi = np.column_stack([theta, phi]).flatten()
        else:
            # Random initialization
            theta = np.random.uniform(0, 2 * np.pi, N)
            # Uniform distribution on sphere: cos(phi) is uniform in [-1, 1]
            cos_phi = np.random.uniform(-1, 1, N)
            phi = np.arccos(cos_phi)
            
            theta_phi = np.column_stack([theta, phi]).flatten()
        
        # Optimize
        try:
            result = minimize(
                _spherical_objective,
                theta_phi,
                jac=_spherical_gradient,
                method='L-BFGS-B',
                bounds=[(0, 2*np.pi), (1e-9, np.pi-1e-9)] * N,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            
            if result.success or result.fun < best_energy:
                # Calculate final energy with high precision
                theta_opt = result.x[0::2]
                phi_opt = result.x[1::2]
                
                # Wrap theta
                theta_opt = theta_opt % (2 * np.pi)
                phi_opt = np.clip(phi_opt, 1e-9, np.pi - 1e-9)
                
                points = _cartesian_from_spherical(theta_opt, phi_opt)
                
                # Calculate precise energy
                precise_energy = _log_energy(points)
                
                if precise_energy < best_energy:
                    best_energy = precise_energy
                    best_theta_phi = result.x.copy()
                    
        except Exception as e:
            pass
        
    # Convert best result back to Cartesian
    if best_theta_phi is not None:
        theta_opt = best_theta_phi[0::2]
        phi_opt = best_theta_phi[1::2]
        
        # Wrap theta
        theta_opt = theta_opt % (2 * np.pi)
        phi_opt = np.clip(phi_opt, 1e-9, np.pi - 1e-9)
        
        best_points = _cartesian_from_spherical(theta_opt, phi_opt)
    else:
        # Fallback
        best_points = _fibonacci_sphere(N)
        
    return best_points

# Helper function for precise energy check (used in construct)
def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(points)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-15:
                return 1e10
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _fibonacci_sphere(n):
    """Generate points on a sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

# EVOLVE-BLOCK-END
