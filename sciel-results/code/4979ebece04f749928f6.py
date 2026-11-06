"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _compute_energy_and_grad(points):
    """
    Compute logarithmic energy and its gradient for points on the sphere.
    Energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t x_i: -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    # Vectorized distance calculation
    # diff[i, j] = x_i - x_j
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # dist2[i, j] = |x_i - x_j|^2
    dist2 = np.sum(diff * diff, axis=-1)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist2, 1.0)
    
    # Energy calculation
    # We need sum_{i<j} log(1/sqrt(dist2)) = -0.5 * sum_{i<j} log(dist2)
    # Using upper triangle
    iu = np.triu_indices(n, k=1)
    # dist = sqrt(dist2)
    # We can compute log(1/dist) = -0.5 * log(dist2)
    # terms = -0.5 * np.log(dist2[iu])
    # But to be precise with math.fsum, let's compute the distances first
    dist = np.sqrt(dist2)
    np.fill_diagonal(dist, 1.0)
    terms = -np.log(dist[iu])
    
    # The task specifies math.fsum for the score. 
    # I will use math.fsum for the energy value returned to ensure it matches 
    # the evaluator's metric exactly.
    energy = math.fsum(terms.tolist())
    
    # Gradient calculation
    # grad[i] = -sum_{j != i} (x_i - x_j) / dist2[i, j]
    # diff[i, j] is x_i - x_j
    # We need to divide each row i by dist2[i, j] for all j != i
    # inv_dist2 = 1 / dist2
    inv_dist2 = 1.0 / dist2
    # Set diagonal to 0 to ignore self-interaction
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = - sum_j diff[i, j] * inv_dist2[i, j]
    # diff is (N, N, 3), inv_dist2 is (N, N)
    # We need to multiply diff[i, j, k] by inv_dist2[i, j] and sum over j
    # This is a matrix multiplication where inv_dist2 is the weight matrix
    # grad = - (diff * inv_dist2[:, :, np.newaxis]).sum(axis=1)
    grad = -np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-15] = 1.0
    return points / norms

def _optimize(points, time_limit):
    """
    Run projected gradient descent with backtracking line search.
    """
    best_energy = _compute_energy_and_grad(points)[0]
    best_points = points.copy()
    
    # Initial step size
    alpha = 0.05
    
    while time.time() < time_limit:
        energy, grad = _compute_energy_and_grad(points)
        
        # If energy is not finite, break
        if not math.isfinite(energy):
            break
            
        if energy < best_energy:
            best_energy = energy
            best_points = points.copy()
            
        # Project gradient to tangent space
        # g_tan = g - (g . x) x
        dot_gx = np.sum(grad * points, axis=1, keepdims=True)
        grad_tan = grad - dot_gx * points
        
        # Check if gradient is small enough to stop
        if np.linalg.norm(grad_tan) < 1e-8:
            break
            
        # Backtracking line search
        # We want to minimize E(x - alpha * grad_tan) projected to sphere
        # Try alpha, reduce by 0.5 if energy increases
        new_points = points - alpha * grad_tan
        new_points = _project_to_sphere(new_points)
        new_energy, _ = _compute_energy_and_grad(new_points)
        
        # If new energy is worse or not finite, reduce alpha
        while new_energy > energy or not math.isfinite(new_energy):
            alpha *= 0.5
            if alpha < 1e-8:
                # Step size too small, likely converged or stuck
                return best_points, best_energy
            new_points = points - alpha * grad_tan
            new_points = _project_to_sphere(new_points)
            new_energy, _ = _compute_energy_and_grad(new_points)
            
        # Accept step
        points = new_points
        # Slightly increase alpha if we succeeded, to speed up
        if alpha < 0.1:
            alpha *= 1.1
            
    return best_points, best_energy

def construct():
    """Place N points on the unit sphere to minimize logarithmic energy."""
    start_time = time.time()
    # Safety margin: 150s limit, stop at 140s
    deadline = start_time + 140.0
    
    best_points = None
    best_energy = float('inf')
    
    # 1. Fibonacci Sphere Initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).astype(np.float64)
    
    # Optimize from Fibonacci start
    time_for_opt = min(deadline - time.time(), 60.0)
    if time_for_opt > 5:
        pts, e = _optimize(points_fib, start_time + time_for_opt)
        if e < best_energy:
            best_energy = e
            best_points = pts
    
    # 2. Random Restarts
    np.random.seed(42)
    while time.time() < deadline - 10.0:
        # Generate random points on sphere
        rand_points = np.random.randn(N, 3)
        rand_points = _project_to_sphere(rand_points)
        
        # Limit time for this restart
        time_for_restart = min(deadline - time.time() - 5.0, 30.0)
        if time_for_restart < 5:
            break
            
        pts, e = _optimize(rand_points, deadline - 5.0)
        
        if e < best_energy:
            best_energy = e
            best_points = pts
            
    if best_points is None:
        best_points = points_fib
        
    # Final polish: Run one more optimization on the best point found
    # to ensure it's fully converged
    if time.time() < deadline - 2.0:
        pts, e = _optimize(best_points, deadline - 1.0)
        if e < best_energy:
            best_energy = e
            best_points = pts
            
    return best_points
# EVOLVE-BLOCK-END
