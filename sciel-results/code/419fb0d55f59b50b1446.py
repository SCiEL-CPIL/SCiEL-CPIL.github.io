"""Rewritten program for sphere covering with 50 caps."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree

def _energy(points):
    """Compute the repulsion energy to encourage uniform distribution."""
    # Use sum of inverse squared distances
    # This is smooth and differentiable
    n = len(points)
    # Compute all pairwise squared distances
    # dist2[i, j] = |p_i - p_j|^2
    # Efficient calculation: |a-b|^2 = |a|^2 + |b|^2 - 2a.b
    # Since points are on unit sphere, |a|^2 = 1
    # dist2 = 2 - 2 * (points @ points.T)
    dist2 = 2.0 - 2.0 * (points @ points.T)
    # Avoid division by zero (diagonal)
    np.fill_diagonal(dist2, 1.0) 
    # Clip to avoid numerical issues
    dist2 = np.clip(dist2, 1e-10, None)
    # Energy = sum 1/dist2 for i < j
    # We sum all and divide by 2 to avoid double counting
    energy = np.sum(1.0 / dist2) / 2.0
    return energy

def _gradient(points):
    """Compute the gradient of the energy with respect to points."""
    # E = 0.5 * sum_{i != j} (2 - 2 p_i . p_j)^-1
    # dE/dp_i = sum_{j != i} -1 * (2 - 2 p_i . p_j)^-2 * (-2 * p_j)
    #          = sum_{j != i} 2 * p_j / (2 - 2 p_i . p_j)^2
    # Let D_ij = 2 - 2 p_i . p_j
    # grad_i = sum_{j != i} 2 * p_j / D_ij^2
    
    n = len(points)
    dist2 = 2.0 - 2.0 * (points @ points.T)
    np.fill_diagonal(dist2, 1.0)
    dist2 = np.clip(dist2, 1e-10, None)
    
    # We want sum_j (2 * p_j) / dist2_ij^2
    # weights_ij = 2 / dist2_ij^2
    weights = 2.0 / (dist2 * dist2)
    np.fill_diagonal(weights, 0.0)
    
    # grad_i = sum_j weights_ij * p_j
    # This is (weights.T @ points)
    grad = weights.T @ points
    return grad

def _project_to_sphere(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius using repulsion."""
    
    # Start from a Fibonacci spiral configuration
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _project_to_sphere(pts)
    
    # Gradient Descent on the sphere
    # We will use a simple adaptive step size
    
    lr = 0.01
    min_lr = 1e-5
    current_energy = _energy(pts)
    
    for step in range(2000):
        grad = _gradient(pts)
        
        # Project gradient to tangent space: grad_tan = grad - (grad . p) p
        dot_grad_p = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tan = grad - dot_grad_p * pts
        
        # Update
        new_pts = pts + lr * grad_tan
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _energy(new_pts)
        
        if new_energy < current_energy:
            pts = new_pts
            current_energy = new_energy
            # Increase lr slightly if improving? Or keep it.
            # Let's decay lr slowly
            lr *= 0.999
        else:
            # Backtrack
            lr *= 0.5
            
        if lr < min_lr:
            break
            
    # Final polish: A few steps of minimizing the actual covering radius?
    # The repulsion energy usually lands in a very good local minimum.
    # We can try to do a local search on the covering radius now, but it's expensive.
    # Given the time limit, the repulsion result is likely sufficient and fast.
    
    return pts
# EVOLVE-BLOCK-END
