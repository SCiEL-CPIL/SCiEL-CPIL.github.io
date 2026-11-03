import numpy as np
import math
from scipy.optimize import minimize

N = 150

def construct():
    # Function to compute energy and gradient for a set of points on the sphere
    # We will use a custom gradient descent with line search on the sphere manifold.
    
    def energy_and_grad(points):
        # points: (N, 3) array of unit vectors
        # Energy: sum_{i<j} log(1/||x_i - x_j||)
        # Grad: sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2 projected to tangent space
        
        dists = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
        # Mask diagonal to avoid division by zero
        np.fill_diagonal(dists, 1.0)
        
        # Energy calculation
        # Use math.fsum for precision if needed, but numpy sum is usually fine for comparison
        # The task specifies math.fsum for the final score, but for optimization, numpy is faster.
        # We will calculate raw energy.
        # To avoid log(0), we already masked diagonal.
        # We sum over i<j.
        iu = np.triu_indices(N, k=1)
        d_pair = dists[iu]
        # Clip to avoid log(0) if points are very close
        d_pair = np.maximum(d_pair, 1e-15)
        E = -np.sum(np.log(d_pair))
        
        # Gradient calculation
        # grad_k = sum_{j != k} (x_k - x_j) / ||x_k - x_j||^2
        # Vectorized:
        # diff[i,j] = x_i - x_j
        # dist_sq[i,j] = ||x_i - x_j||^2
        # term[i,j] = (x_i - x_j) / dist_sq[i,j]
        # grad_i = sum_{j != i} term[i,j]
        
        diff = points[:, np.newaxis] - points[np.newaxis, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0) # Avoid div by 0
        
        # We want term[i,j] = diff[i,j] / dist_sq[i,j]
        # But we only sum j != i.
        # Create a mask for j != i
        mask = ~np.eye(N, dtype=bool)
        dist_sq_safe = np.where(mask, dist_sq, 1.0)
        
        # term[i,j,k] = diff[i,j,k] / dist_sq[i,j]
        terms = diff / dist_sq_safe[:, :, np.newaxis]
        # Zero out diagonal terms
        terms[:, np.arange(N), np.arange(N)] = 0.0
        
        # Sum over j (axis 1)
        grad = np.sum(terms, axis=1)
        
        # Project gradient to tangent space
        # grad_tangent = grad - (grad . x) x
        dot = np.sum(grad * points, axis=1)
        grad_tangent = grad - dot[:, np.newaxis] * points
        
        return E, grad_tangent

    def optimize_from(points):
        # Gradient Descent with Backtracking Line Search
        current_points = points.copy()
        current_energy, current_grad = energy_and_grad(current_points)
        
        step_size = 0.01
        max_iterations = 5000
        min_step = 1e-8
        
        for _ in range(max_iterations):
            if np.linalg.norm(current_grad) < 1e-6:
                break
            
            # Proposed step
            new_points = current_points - step_size * current_grad
            # Normalize
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            new_points = new_points / norms
            
            new_energy, new_grad = energy_and_grad(new_points)
            
            # Backtracking line search
            if new_energy < current_energy:
                current_points = new_points
                current_energy = new_energy
                current_grad = new_grad
                # Increase step size slightly if improvement is good
                step_size = min(step_size * 1.05, 0.1)
            else:
                step_size *= 0.5
                if step_size < min_step:
                    break
                    
        return current_points, current_energy

    # Multi-start strategy
    best_energy = np.inf
    best_points = None
    
    # Use a few random initializations
    # 1. Fibonacci (deterministic, good spread)
    # 2. Random normal
    # 3. Random uniform on sphere
    
    inits = []
    
    # Init 1: Fibonacci
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(fib_points)
    
    # Init 2: Random Normal
    np.random.seed(42)
    for _ in range(3):
        rand_points = np.random.randn(N, 3)
        rand_points /= np.linalg.norm(rand_points, axis=1, keepdims=True)
        inits.append(rand_points)
        
    for init_points in inits:
        opt_points, opt_energy = optimize_from(init_points)
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_points = opt_points
            
    return best_points
