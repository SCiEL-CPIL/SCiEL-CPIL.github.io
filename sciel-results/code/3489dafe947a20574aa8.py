"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms

def _covering_radius(points):
    """Compute the covering radius in degrees for a set of points on the unit sphere.
    Uses the fact that the covering radius is the maximum over all pairs of the 
    angular distance to the nearest point, which can be approximated by checking
    the circumcenters of Delaunay triangles or using a dense sampling.
    For efficiency, we use a dense sampling approach with adaptive refinement."""
    # Generate a dense set of test points on the sphere
    # Use a Fibonacci spiral with many more points for good coverage
    n_test = 50000
    i = np.arange(n_test) + 0.5
    z = 1.0 - 2.0 * i / n_test
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # For each test point, find the nearest configuration point
    # Compute dot products (cosines of angles)
    # Points are already normalized
    dots = test_pts @ points.T  # shape (n_test, N)
    # Angular distance = arccos(dot), but we want to maximize the min distance
    # So for each test point, find max dot (nearest point)
    max_dots = np.max(dots, axis=1)
    # Clip to [-1, 1] for numerical stability
    max_dots = np.clip(max_dots, -1.0, 1.0)
    angles = np.arccos(max_dots)
    max_angle = np.max(angles)
    return np.degrees(max_angle)

def _soft_max_objective(params, temperature=100.0):
    """Differentiable approximation of the covering radius using soft-max.
    params: flattened array of N*3 coordinates.
    Returns: scalar objective value (approximate covering radius in radians)."""
    points = params.reshape(N, 3)
    points = _normalize(points)
    
    # Sample test points
    n_test = 10000
    i = np.arange(n_test) + 0.5
    z = 1.0 - 2.0 * i / n_test
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute cosines of angles
    dots = test_pts @ points.T  # (n_test, N)
    dots = np.clip(dots, -1.0, 1.0)
    
    # For each test point, we want the angle to the nearest point
    # angle = arccos(max_dot)
    # We use soft-max to approximate max_dot differentiably
    # soft_max(x) = (1/temperature) * log(sum(exp(temperature * x)))
    # But we want max of dots, which is equivalent to min of angles
    # Actually, we want to maximize the minimum angle, so we want to minimize the maximum angle
    # max_angle = max over test points of arccos(max_dot)
    # Let's compute soft_max of dots for each test point
    scaled_dots = temperature * dots
    # Subtract max for numerical stability
    max_scaled = np.max(scaled_dots, axis=1, keepdims=True)
    exp_dots = np.exp(scaled_dots - max_scaled)
    sum_exp = np.sum(exp_dots, axis=1, keepdims=True)
    soft_max_dots = (max_scaled + np.log(sum_exp)) / temperature
    
    soft_max_dots = np.clip(soft_max_dots, -1.0, 1.0)
    angles = np.arccos(soft_max_dots)
    return np.max(angles)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Try multiple restarts with different initializations and optimization
    np.random.seed(42)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
            
        if restart == 0:
            # Use the Fibonacci spiral as starting point
            current_points = points.copy()
        else:
            # Random perturbation of the Fibonacci spiral
            current_points = points + np.random.randn(N, 3) * 0.1
            current_points = _normalize(current_points)
        
        # Gradient descent with dynamic temperature
        params = current_points.flatten().copy()
        temperature = 50.0
        
        for step in range(500):
            if time.time() > deadline - 10.0:
                break
                
            # Compute objective
            obj = _soft_max_objective(params, temperature)
            
            # Compute gradient via finite differences (expensive but simple)
            # Actually, let's use a more efficient approach: compute gradient analytically
            # For now, use numerical gradient with a few points
            epsilon = 1e-4
            grad = np.zeros_like(params)
            
            # Sample a subset of test points for gradient computation
            n_grad_test = 2000
            i_g = np.arange(n_grad_test) + 0.5
            z_g = 1.0 - 2.0 * i_g / n_grad_test
            phi_g = np.pi * (1.0 + np.sqrt(5.0)) * i_g
            r_g = np.sqrt(np.maximum(0, 1.0 - z_g * z_g))
            test_pts_g = np.column_stack([r_g * np.cos(phi_g), r_g * np.sin(phi_g), z_g])
            
            p = params.reshape(N, 3)
            p = _normalize(p)
            
            # For each test point, find the nearest point and compute gradient contribution
            dots_g = test_pts_g @ p.T  # (n_grad_test, N)
            dots_g = np.clip(dots_g, -1.0, 1.0)
            
            # Soft max
            scaled = temperature * dots_g
            max_scaled = np.max(scaled, axis=1, keepdims=True)
            exp_d = np.exp(scaled - max_scaled)
            sum_exp = np.sum(exp_d, axis=1, keepdims=True)
            weights = exp_d / sum_exp  # (n_grad_test, N)
            
            soft_max_dots_g = (max_scaled + np.log(sum_exp)) / temperature
            soft_max_dots_g = np.clip(soft_max_dots_g, -1.0, 1.0)
            angles_g = np.arccos(soft_max_dots_g)
            
            # The objective is max over test points of angle
            # For gradient, we can use the weighted average as an approximation
            # d(angle)/d(dot) = -1/sqrt(1-dot^2)
            d_angle_ddot = -1.0 / np.sqrt(np.maximum(1e-12, 1.0 - soft_max_dots_g ** 2))
            
            # Gradient of soft_max_dot with respect to dot_ij = weights[i,j]
            # Gradient of dot_ij with respect to p_jk = test_pts_g[i,k]
            # So grad_p[j,k] = sum_i weights[i,j] * d_angle_ddot[i] * test_pts_g[i,k]
            # Note: d_angle_ddot is negative, so this pushes points away from test points
            
            # Actually, we want to minimize the max angle, so we want to move points
            # to reduce the angle. The gradient of angle w.r.t. dot is negative,
            # meaning increasing dot decreases angle. So we want to increase dots,
            # which means moving points closer to test points.
            
            # grad_p[j] = sum_i weights[i,j] * d_angle_ddot[i] * test_pts_g[i]
            # But d_angle_ddot is negative, so this is a descent direction for angle
            
            grad_p = (weights.T * d_angle_ddot) @ test_pts_g  # (N, 3)
            
            # Project gradient onto tangent space (remove radial component)
            grad_p -= np.sum(grad_p * p, axis=1, keepdims=True) * p
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad_p)
            if grad_norm > 1e-12:
                grad_p /= grad_norm
            
            grad = grad_p.flatten()
            
            # Adaptive learning rate
            lr = 0.01 / (1.0 + step * 0.01)
            
            # Update
            params -= lr * grad
            
            # Periodically normalize
            if step % 10 == 0:
                p_check = params.reshape(N, 3)
                p_check = _normalize(p_check)
                params = p_check.flatten()
        
        # Evaluate final configuration
        final_points = _normalize(params.reshape(N, 3))
        final_radius = _covering_radius(final_points)
        
        if final_radius < best_radius:
            best_radius = final_radius
            best_points = final_points.copy()
    
    return best_points
# EVOLVE-BLOCK-END
