"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

# Hard deadline with safety margin
_DEADLINE = 145.0

def _fast_log_energy(pts):
    """
    Compute sum_{i<j} log(1/|xi-xj|) efficiently.
    Uses np.sum for speed. The evaluator uses math.fsum, but for optimization 
    purposes, np.sum is a sufficient proxy and much faster.
    Returns +inf if any distance < 1e-9.
    """
    n = pts.shape[0]
    # Compute squared distances to avoid sqrt in most cases? 
    # log(1/d) = -0.5 * log(d^2). 
    # d^2 = |xi - xj|^2 = 2 - 2*xi.dot(xj)
    # This is much faster than computing full diff matrix and norms.
    
    # Gram matrix G = pts @ pts.T
    # G[i,j] = xi . xj
    # Since pts are normalized, G[i,i] = 1.
    G = pts @ pts.T
    
    # d^2 = 2 - 2*G
    # We need upper triangle i < j
    iu = np.triu_indices(n, k=1)
    d_sq = 2.0 - 2.0 * G[iu]
    
    # Check for close points
    if np.any(d_sq < 1e-18): # 1e-9 squared
        return 1e100
        
    # Energy = sum log(1/sqrt(d_sq)) = -0.5 * sum log(d_sq)
    # Clip d_sq to avoid log(0) or log(negative due to numerical error)
    # d_sq should be positive. Min valid d is 1e-9 -> d_sq is 1e-18.
    # Numerical errors might make d_sq slightly negative for very close points.
    d_sq = np.maximum(d_sq, 1e-15)
    
    return -0.5 * np.sum(np.log(d_sq))

def _compute_gradient(pts):
    """
    Compute the gradient of the log-energy with respect to the points on the sphere.
    E = sum_{i<j} log(1/|xi - xj|) = - sum_{i<j} log(|xi - xj|)
    
    Gradient of log(1/|xi - xj|) w.r.t xi:
    d/dxi log(1/|xi - xj|) = d/dxi (-log|xi - xj|)
    = - (1/|xi - xj|) * d/dxi |xi - xj|
    = - (1/|xi - xj|) * (xi - xj)/|xi - xj|
    = -(xi - xj)/|xi - xj|^2
    = (xj - xi)/|xi - xj|^2
    
    So, grad_i E = sum_{j != i} (xj - xi) / |xi - xj|^2
    
    Note: This is the gradient in R^3. To perform gradient descent on the sphere,
    we project the negative gradient onto the tangent plane at xi:
    v_i = -grad_i E
    v_tangent = v_i - (v_i . xi) xi
    Then update xi <- xi + step * v_tangent, and normalize.
    """
    n = pts.shape[0]
    # G = pts @ pts.T
    G = pts @ pts.T
    
    # d_sq[i,j] = |xi - xj|^2 = 2 - 2*G[i,j]
    d_sq = 2.0 - 2.0 * G
    
    # Set diagonal to 1 to avoid division by zero (we won't use it)
    np.fill_diagonal(d_sq, 1.0)
    
    # We need sum_{j != i} (xj - xi) / d_sq[i,j]
    # Let W[i,j] = 1/d_sq[i,j] for i != j, 0 otherwise.
    # grad_i = sum_j W[i,j] * xj - sum_j W[i,j] * xi
    #        = (W @ pts)[i] - xi * sum_j W[i,j]
    
    W = 1.0 / d_sq
    np.fill_diagonal(W, 0.0)
    
    # Sum of weights for each i
    w_sum = W.sum(axis=1)
    
    # (W @ pts) gives sum_j W[i,j] xj for each i
    weighted_pts = W @ pts
    
    # grad_i = weighted_pts[i] - xi * w_sum[i]
    # This is the gradient of E. We want to minimize E, so we move in -grad direction.
    # But first, let's compute the raw gradient.
    grad = weighted_pts - pts * w_sum[:, np.newaxis]
    
    return grad

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _fibonacci_sphere(n, offset=0.0):
    """Generate n Fibonacci spiral points on unit sphere with angular offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _gradient_descent_step(pts, step_size):
    """
    Perform one gradient descent step on the sphere.
    """
    grad = _compute_gradient(pts)
    
    # Project negative gradient onto tangent plane
    # v = -grad
    v = -grad
    
    # Project v onto tangent plane at xi: v_t = v - (v . xi) xi
    # v . xi for each i
    v_dot_xi = np.einsum('ij,ij->i', v, pts)
    v_tangent = v - pts * v_dot_xi[:, np.newaxis]
    
    # Update
    new_pts = pts + step_size * v_tangent
    
    # Normalize
    new_pts = _normalize(new_pts)
    
    return new_pts

def construct():
    start_time = time.time()
    
    best_pts = None
    best_energy = float('inf')
    
    # Multi-start strategy
    # We will try several initial configurations:
    # 1. Standard Fibonacci
    # 2. Fibonacci with different offsets
    # 3. Random points
    
    initial_configs = []
    initial_configs.append(_fibonacci_sphere(N))
    for offset in [0.1, 0.5, 1.0, 2.0]:
        initial_configs.append(_fibonacci_sphere(N, offset))
    
    # Add a few random starts
    for _ in range(3):
        vecs = np.random.normal(size=(N, 3))
        initial_configs.append(_normalize(vecs))
    
    for init_pts in initial_configs:
        if time.time() > _DEADLINE:
            break
            
        current_pts = init_pts.copy()
        current_energy = _fast_log_energy(current_pts)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = current_pts.copy()
            
        # Gradient Descent Phase
        # Start with a moderate step size, decay it over time
        step_size = 0.05
        min_step_size = 1e-6
        
        # We will run gradient descent for a fixed number of iterations or until time limit
        # Since we have multiple starts, we should limit time per start.
        # Let's say we allow 10s per start? No, 140s total / 7 starts ~ 20s each.
        # But later starts might be better. 
        # Better: run until convergence or time limit, but check global time.
        
        # Adaptive step size
        # If energy increases, reduce step size.
        
        for iter in range(5000): # Cap iterations to avoid infinite loop if time check fails
            if time.time() > _DEADLINE:
                break
                
            new_pts = _gradient_descent_step(current_pts, step_size)
            new_energy = _fast_log_energy(new_pts)
            
            if new_energy < current_energy:
                current_pts = new_pts
                current_energy = new_energy
                # Decay step size slowly
                step_size *= 0.999
            else:
                # If energy increased, reduce step size significantly
                step_size *= 0.5
                
            if step_size < min_step_size:
                break
                
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = current_pts.copy()
            
        # Occasionally, if we are stuck in a local minimum, we could do a random kick and restart?
        # But gradient descent is deterministic. 
        # The multi-start handles the global search aspect.
        
    # Final verification
    if best_pts is None:
        return _fibonacci_sphere(N)
        
    # Check validity
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    if np.any(d < 1e-9):
        return _fibonacci_sphere(N)
        
    return best_pts
# EVOLVE-BLOCK-END
