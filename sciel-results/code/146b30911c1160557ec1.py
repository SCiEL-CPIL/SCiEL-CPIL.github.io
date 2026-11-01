"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _energy_and_grad(x):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient w.r.t. x (in R^{N*3}).
    E = - sum_{i<j} log(|x_i - x_j|)
    grad_i E = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    x = x.reshape(N, 3)
    # Pairwise differences: x_i - x_j for all i, j
    # diff[i, j] = x[i] - x[j]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    dists = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum_{i<j} log(1/d) = -0.5 * sum_{i!=j} log(d)
    # Using fsum for accuracy on the upper triangle
    iu = np.triu_indices(N, k=1)
    d_upper = dists[iu]
    E = -math.fsum(np.log(d_upper).tolist())
    
    # Gradient: grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] / dists[i,j]^2, summed over j != i
    inv_d2 = np.zeros_like(dists)
    np.fill_diagonal(inv_d2, 0.0)
    inv_d2 = 1.0 / (dists * dists)
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = sum_j diff[i,j] * inv_d2[i,j]
    grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    return E, grad.flatten()


def _project_to_sphere(x):
    """Project points onto unit sphere."""
    x = x.reshape(N, 3)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms


def _halton_sequence(n, dims):
    """Generate Halton sequence points in [0,1)^dims."""
    def halton_index(i, base):
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    points = np.zeros((n, dims))
    for d in range(dims):
        for i in range(n):
            points[i, d] = halton_index(i + 1, primes[d])
    return points


def _halton_init():
    """Initialize points using Halton sequence mapped to sphere."""
    pts = _halton_sequence(N, 3)
    # Map to sphere: use spherical coordinates
    # z in [-1, 1], theta in [0, 2pi)
    z = 2.0 * pts[:, 0] - 1.0
    theta = 2.0 * np.pi * pts[:, 1]
    r = np.sqrt(np.maximum(1.0 - z*z, 0.0))
    x = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    # Normalize
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms


def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # 140s safety margin before 150s kill
    
    # Initialize with Halton sequence
    x0 = _halton_init().flatten()
    
    # First, do a projected gradient descent to get close to a good local min
    # Then use L-BFGS-B with equality constraint (sphere) via penalty or projection
    
    # Strategy: Use L-BFGS-B on the sphere by parameterizing or using projection
    # Since L-BFGS-B doesn't handle equality constraints, we use a penalty approach
    # or we can use scipy's SLSQP with equality constraints
    
    # Better: use the fact that we can project after each step
    # We'll use a custom projected gradient method followed by L-BFGS on tangent space
    
    # Step 1: Projected gradient descent for initial refinement
    best_x = x0.copy()
    best_E, _ = _energy_and_grad(best_x)
    
    # Simple projected gradient descent
    lr = 0.01
    for step in range(500):
        if time.time() > deadline - 30.0:
            break
        E, g = _energy_and_grad(best_x)
        if E < best_E:
            best_E = E
        # Project gradient to tangent space
        x_reshaped = best_x.reshape(N, 3)
        g_reshaped = g.reshape(N, 3)
        # Tangent projection: g_tan = g - (g . x) x
        g_dot_x = np.sum(g_reshaped * x_reshaped, axis=1, keepdims=True)
        g_tan = g_reshaped - g_dot_x * x_reshaped
        # Update
        best_x = best_x - lr * g_tan.flatten()
        # Project to sphere
        best_x = _project_to_sphere(best_x).flatten()
    
    # Step 2: Use L-BFGS-B with a penalty for sphere constraint
    # Or use SLSQP with equality constraints
    
    # Let's use a more direct approach: optimize on the sphere using
    # the exponential map / retraction
    
    # Actually, let's just use scipy.optimize with the constraint that ||x_i|| = 1
    # We'll use SLSQP which supports equality constraints
    
    # Define constraints
    def sphere_constraint(x):
        x_reshaped = x.reshape(N, 3)
        return np.sum(x_reshaped * x_reshaped, axis=1) - 1.0
    
    # Use L-BFGS-B with penalty method
    # Or better: use the fact that we can do line search with projection
    
    # Let's do a more careful optimization using the tangent space formulation
    # We'll alternate between:
    # 1. Compute gradient in R^3
    # 2. Project to tangent space
    # 3. Take step in tangent space
    # 4. Retract to sphere (normalize)
    
    # Use L-BFGS on the reparameterized problem
    # Actually, let's just do a robust projected gradient with adaptive step size
    
    best_E, _ = _energy_and_grad(best_x)
    best_x_copy = best_x.copy()
    
    # Adaptive projected gradient descent with backtracking
    lr = 0.01
    for it in range(3000):
        if time.time() > deadline - 15.0:
            break
        E, g = _energy_and_grad(best_x)
        if E < best_E:
            best_E = E
            best_x_copy = best_x.copy()
        
        x_reshaped = best_x.reshape(N, 3)
        g_reshaped = g.reshape(N, 3)
        g_dot_x = np.sum(g_reshaped * x_reshaped, axis=1, keepdims=True)
        g_tan = g_reshaped - g_dot_x * x_reshaped
        
        # Backtracking line search
        alpha = lr
        E_new = E
        x_new = best_x.copy()
        for _ in range(20):
            x_trial = best_x - alpha * g_tan.flatten()
            x_trial = _project_to_sphere(x_trial).flatten()
            E_trial, _ = _energy_and_grad(x_trial)
            if E_trial < E_new:
                E_new = E_trial
                x_new = x_trial
                break
            alpha *= 0.5
        best_x = x_new
        lr = alpha * 1.5 if E_new < E else alpha * 0.5
        lr = min(lr, 0.1)
        lr = max(lr, 1e-6)
    
    # Final polish with L-BFGS using a penalty method
    if time.time() < deadline - 5.0:
        def penalized_energy(x, mu=1e4):
            x_reshaped = x.reshape(N, 3)
            E, g = _energy_and_grad(x)
            # Penalty for deviation from unit sphere
            norms_sq = np.sum(x_reshaped * x_reshaped, axis=1)
            penalty = mu * np.sum((norms_sq - 1.0)**2)
            # Gradient of penalty
            pen_grad = 4.0 * mu * (norms_sq - 1.0) * x_reshaped
            return E + penalty, (g + pen_grad).flatten()
        
        # Use L-BFGS-B on the penalized problem
        try:
            result = minimize(
                penalized_energy,
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_E:
                best_x = result.x
                best_E = _energy_and_grad(best_x)[0]
                best_x = _project_to_sphere(best_x).flatten()
                best_E = _energy_and_grad(best_x)[0]
        except Exception:
            pass
    
    # Return the best result
    best_x = _project_to_sphere(best_x).flatten()
    return best_x.reshape(N, 3)

# EVOLVE-BLOCK-END
