import numpy as np
import math
from scipy.optimize import minimize

N = 150

def compute_energy_and_grad(points):
    # points shape (N, 3)
    # Compute pairwise distances
    # diff[i, j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0) # Avoid div by zero for gradient calc, but we need 0 on diagonal for sum
    
    # Calculate Energy
    # We only need i < j
    triu = np.triu_indices(N, k=1)
    d_pair = np.sqrt(dist_sq[triu])
    # Handle very small distances to avoid log(0)
    d_pair = np.maximum(d_pair, 1e-15)
    energy = -np.sum(np.log(d_pair))
    
    # Calculate Gradient
    # grad_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # This is equivalent to: - sum_j (x_i - x_j) / |x_i - x_j|^2
    # Note: for j=i, term is 0/1 = 0.
    # So we can just sum over all j.
    inv_dist_sq = 1.0 / dist_sq
    # forces_raw[i, j] = (x_i - x_j) / |x_i - x_j|^2
    # grad_i = - sum_j forces_raw[i, j]
    # But we must be careful: the term for j=i is 0.
    # So grad = - np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad

def objective_and_grad(flat_points):
    points = flat_points.reshape(N, 3)
    # Penalty for sphere constraint
    # P = M * sum (|x_i|^2 - 1)^2
    M = 1000.0
    norms_sq = np.sum(points * points, axis=1)
    penalty = M * np.sum((norms_sq - 1.0) ** 2)
    
    energy, grad_energy = compute_energy_and_grad(points)
    
    # Gradient of penalty
    # d/dx_i (M (|x_i|^2 - 1)^2) = M * 2 (|x_i|^2 - 1) * 2 x_i = 4 M x_i (|x_i|^2 - 1)
    grad_penalty = 4.0 * M * points * (norms_sq - 1.0)[:, np.newaxis]
    
    total_obj = energy + penalty
    total_grad = grad_energy + grad_penalty
    
    return total_obj, total_grad

def construct():
    best_energy = np.inf
    best_points = None
    
    # Initial guess 1: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    candidates = [init_fib]
    
    # Add some random initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        # Random points on sphere
        v = rng.normal(size=(N, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        candidates.append(v)
        
    for init in candidates:
        # Flatten for L-BFGS-B
        x0 = init.flatten()
        
        # Run L-BFGS-B
        res = minimize(
            objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-9}
        )
        
        # Extract result
        points = res.x.reshape(N, 3)
        
        # Normalize to sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        # Calculate true energy (without penalty)
        # Reuse logic from compute_energy_and_grad but just energy
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        triu = np.triu_indices(N, k=1)
        d_pair = np.sqrt(dist_sq[triu])
        d_pair = np.maximum(d_pair, 1e-15)
        current_energy = -np.sum(np.log(d_pair))
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_points = points.copy()
            
    return best_points
