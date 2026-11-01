"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize

# Deadline guard: 150s limit, leave 5s buffer for overhead
DEADLINE = time.time() + 145.0

def _log_energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient for points on the unit sphere.
    Energy: sum_{i<j} log(1 / ||x_i - x_j||)
    Gradient: For each i, grad_i = sum_{j != i} (x_i - x_j) / (2 * ||x_i - x_j||^2)
    Note: The gradient is in R^3. We project it onto the tangent space in the optimizer step
    or use a penalty method. However, L-BFGS-B works on unconstrained variables.
    A common trick is to parameterize points or use a penalty.
    Simpler: Use the fact that we want to minimize energy on the sphere.
    We can use the gradient of the energy function defined on R^3, but we must ensure
    the points stay on the sphere.
    
    Actually, the standard approach for sphere constraints with L-BFGS-B is difficult 
    without reparameterization. 
    Alternative: Use spherical coordinates? No, singularities.
    Alternative: Use the fact that the energy is invariant to rotation, and use 
    a penalty for ||x||^2 - 1.
    
    Let's use a different approach: 
    The gradient of E = sum log(1/d_ij) w.r.t x_i is:
    dE/dx_i = sum_{j != i} (x_i - x_j) / (2 * d_ij^2)
    
    If we simply apply this gradient, the points will drift off the sphere.
    We can project the gradient onto the tangent space: 
    grad_tangent_i = grad_i - (grad_i . x_i) x_i
    Then take a step in the tangent direction and re-normalize.
    This is essentially a gradient descent on the manifold.
    L-BFGS-B does not support manifold constraints directly.
    
    Instead, let's use a parameterization that guarantees the sphere:
    Use 2D coordinates (theta, phi) or use the "gauge" trick?
    Actually, for N=150, we can use a simple gradient descent with re-normalization 
    and line search, or use scipy.optimize.minimize with a penalty.
    
    However, a more robust method for this specific problem is often 
    "Riesz s-energy" minimization via gradient descent on the sphere.
    
    Let's implement a custom gradient descent step function for minimize? 
    No, minimize expects a function f(x) and optionally jac(x).
    
    Let's try the penalty method:
    f(x) = Energy(x) + lambda * sum(||x_i||^2 - 1)^2
    But Energy is not defined for non-sphere points in the same way? 
    The formula log(1/||x_i-x_j||) is defined for any x_i != x_j.
    
    Let's stick to the tangent space gradient descent implemented manually 
    with a simple line search, as it's more reliable for the sphere constraint 
    than penalty methods which can suffer from ill-conditioning.
    
    Actually, let's look at the "Inspired" solutions. They often use 
    scipy.optimize.minimize with method='L-BFGS-B' and bounds? No, sphere is not box.
    
    Let's use the parameterization:
    Each point is represented by 2 angles (theta, phi).
    x = [sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)]
    This maps R^2 -> S^2.
    We need to compute the gradient w.r.t theta and phi.
    
    Let x_i = f(theta_i, phi_i).
    dE/dtheta_i = (dE/dx_i) . (dx_i/dtheta_i)
    dE/dphi_i = (dE/dx_i) . (dx_i/dphi_i)
    
    dx/dtheta = [cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta)]
    dx/dphi = [-sin(theta)sin(phi), sin(theta)cos(phi), 0]
    
    This is computationally heavier but avoids constraint issues.
    N=150 points -> 300 variables.
    
    Let's implement this.
    """
    # pts is (N, 2) containing (theta, phi)
    # Convert to cartesian for distance calculation
    theta = pts[:, 0]
    phi = pts[:, 1]
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    
    x = sin_t * cos_p
    y = sin_t * sin_p
    z = cos_t
    
    # Compute distances
    # diff[i, j] = x_i - x_j
    # d_ij = ||x_i - x_j||
    
    # Vectorized distance calculation
    # x, y, z are (N,)
    # We need pairwise distances.
    
    # Using broadcasting
    # x_i - x_j
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dz = z[:, None] - z[None, :]
    
    d_sq = dx**2 + dy**2 + dz**2
    # Avoid division by zero
    d_sq[diag_indices] = 1.0 # Just a placeholder, we will mask it
    d = np.sqrt(d_sq)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # Let's compute the full sum and divide by 2.
    # log(1/d) = -log(d)
    
    # To avoid log(0), we clipped d_sq to 1e-12 earlier? 
    # Let's use np.where to handle small distances.
    d_safe = np.where(d < 1e-12, 1e-12, d)
    
    log_d = np.log(d_safe)
    # Sum over i != j
    total_log_d = np.sum(log_d[mask])
    energy = -0.5 * total_log_d
    
    # Gradient
    # dE/dx_i = sum_{j != i} (x_i - x_j) / (2 * d_ij^2)
    # Note: The factor 1/2 comes from the 1/2 in the sum definition if we sum over all i,j.
    # Let's re-verify.
    # E = sum_{i<j} -log(d_ij)
    # dE/dx_i = sum_{j != i, j > i} (x_i - x_j)/d_ij^2 + sum_{j != i, j < i} (x_i - x_j)/d_ij^2 ?
    # No.
    # E = 1/2 sum_{i != j} -log(d_ij)
    # dE/dx_i = 1/2 sum_{j != i} - (1/d_ij) * d(d_ij)/dx_i
    # d(d_ij)/dx_i = (x_i - x_j) / d_ij
    # So dE/dx_i = 1/2 sum_{j != i} - (x_i - x_j) / d_ij^2
    # Wait, log(1/d) = -log(d).
    # d(-log(d))/dx = - (1/d) * (x_i - x_j)/d = - (x_i - x_j)/d^2.
    # So dE/dx_i = 1/2 sum_{j != i} - (x_i - x_j) / d_ij^2
    # This is the gradient in Cartesian space.
    
    # Let's compute G_i = sum_{j != i} (x_i - x_j) / d_ij^2
    # Then grad_x_i = -0.5 * G_i
    
    # d_sq is (N, N)
    # We need to sum (x_i - x_j) / d_sq[i, j] over j.
    # dx is (N, N, 1) if we expand dims.
    
    # Let's construct the vector field.
    # inv_d_sq = 1 / d_sq
    # Set diagonal to 0
    inv_d_sq = np.zeros_like(d_sq)
    inv_d_sq[mask] = 1.0 / d_sq[mask]
    
    # grad_x[i] = sum_j (x_i - x_j) * inv_d_sq[i, j]
    # dx[i, j] = x_i - x_j
    # We can do this with einsum or matrix multiplication.
    # grad_x = dx * inv_d_sq[..., None] ? No, dx is (N, N, 3) effectively.
    
    # Let's use the components x, y, z.
    # grad_x_comp[i] = sum_j (x[i] - x[j]) * inv_d_sq[i, j]
    # = x[i] * sum_j inv_d_sq[i, j] - sum_j x[j] * inv_d_sq[i, j]
    
    # Let's compute S1_i = sum_j inv_d_sq[i, j]
    # Let's compute S2_x_i = sum_j x[j] * inv_d_sq[i, j]
    # Similarly for y, z.
    
    S1 = np.sum(inv_d_sq, axis=1)
    S2_x = inv_d_sq @ x
    S2_y = inv_d_sq @ y
    S2_z = inv_d_sq @ z
    
    grad_x_cart = x * S1 - S2_x
    grad_y_cart = y * S1 - S2_y
    grad_z_cart = z * S1 - S2_z
    
    # This is G_i.
    # grad_E_cart_x = -0.5 * grad_x_cart
    # grad_E_cart_y = -0.5 * grad_y_cart
    # grad_E_cart_z = -0.5 * grad_z_cart
    
    # Now transform to (theta, phi)
    # dE/dtheta = dE/dx * dx/dtheta + dE/dy * dy/dtheta + dE/dz * dz/dtheta
    # dx/dtheta = cos_t * cos_p
    # dy/dtheta = cos_t * sin_p
    # dz/dtheta = -sin_t
    
    # dE/dphi = dE/dx * dx/dphi + dE/dy * dy/dphi + dE/dz * dz/dphi
    # dx/dphi = -sin_t * sin_p
    # dy/dphi = sin_t * cos_p
    # dz/dphi = 0
    
    # Note: The gradient in Cartesian is for the energy E.
    # g_x = -0.5 * grad_x_cart, etc.
    
    g_x = -0.5 * grad_x_cart
    g_y = -0.5 * grad_y_cart
    g_z = -0.5 * grad_z_cart
    
    grad_theta = g_x * (cos_t * cos_p) + g_y * (cos_t * sin_p) + g_z * (-sin_t)
    grad_phi = g_x * (-sin_t * sin_p) + g_y * (sin_t * cos_p)
    
    # Handle phi singularity at poles.
    # If sin_t is very small, grad_phi is ill-defined or zero.
    # We can just let it be, the optimizer should handle it if initialized away from poles.
    
    return energy, np.column_stack((grad_theta, grad_phi))

def construct():
    # Start time
    t_start = time.time()
    
    # Initial guess: Fibonacci spiral converted to (theta, phi)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    # theta = arccos(z)
    theta_init = np.arccos(z)
    phi_init = np.pi * (1.0 + np.sqrt(5.0)) * i
    
    # Convert to cartesian to check
    # But we need to return cartesian.
    
    # Let's do a few random restarts if time permits.
    # For now, one good run.
    
    # Define function for scipy
    def fun_and_jac(x):
        e, g = _log_energy_and_grad(x)
        return e, g
        
    # Options
    options = {
        'maxiter': 1000,
        'ftol': 1e-15,
        'gtol': 1e-15,
        'maxls': 50
    }
    
    # Run optimization
    # Note: minimize with jac=True expects fun and jac to be separate or a tuple.
    # We passed a function returning tuple, so we need to use method that supports it?
    # L-BFGS-B supports jac.
    
    try:
        res = minimize(fun_and_jac, np.column_stack((theta_init, phi_init)), 
                       method='L-BFGS-B', 
                       jac=True,
                       options=options)
    except Exception as e:
        # Fallback to initial guess
        pass
    else:
        if res.success or res.fun < _log_energy_and_grad(np.column_stack((theta_init, phi_init)))[0]:
            theta_opt, phi_opt = res.x
        else:
            theta_opt, phi_opt = theta_init, phi_init
            
    # Convert back to cartesian
    sin_t = np.sin(theta_opt)
    cos_t = np.cos(theta_opt)
    sin_p = np.sin(phi_opt)
    cos_p = np.cos(phi_opt)
    
    x = sin_t * cos_p
    y = sin_t * sin_p
    z = cos_t
    
    pts = np.column_stack((x, y, z))
    
    # Normalize to ensure unit sphere (numerical stability)
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, None]
    
    # Check time
    if time.time() > DEADLINE:
        return pts
        
    # Optional: Second restart with random perturbation if time allows
    # For now, just return the best found.
    
    return pts

# EVOLVE-BLOCK-END
