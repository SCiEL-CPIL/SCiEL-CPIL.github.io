import numpy as np
import math

N = 150

def compute_energy(points):
    """
    Compute the logarithmic energy for points on the unit sphere.
    points: (N, 3) array
    """
    # Pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Get upper triangle indices (i < j)
    triu = np.triu_indices(N, k=1)
    d_pair = np.sqrt(dist_sq[triu])
    
    # Avoid log(0)
    d_pair = np.maximum(d_pair, 1e-15)
    
    energy = -np.sum(np.log(d_pair))
    return energy

def compute_riemannian_gradient(points):
    """
    Compute the Riemannian gradient of the energy on the sphere.
    points: (N, 3) array
    Returns: (N, 3) array of tangent vectors
    """
    # Euclidean gradient: grad_i = - sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Mask diagonal to avoid division by zero
    np.fill_diagonal(dist_sq, 1.0)
    
    # forces_raw[i, j] = (x_i - x_j) / ||x_i - x_j||^2
    # We want sum over j for each i
    inv_dist_sq = 1.0 / dist_sq
    # Multiply each row vector by its corresponding inverse distance squared
    # diff is (N, N, 3), inv_dist_sq is (N, N)
    forces_raw = diff * inv_dist_sq[:, :, np.newaxis]
    
    # Sum over j (axis 1) to get gradient for each i
    # grad_euclidean[i] = - sum_j forces_raw[i, j]
    grad_euclidean = -np.sum(forces_raw, axis=1)
    
    # Project to tangent space: g_tan = g_eucl - (g_eucl . x_i) x_i
    # dot products
    dots = np.sum(grad_euclidean * points, axis=1)
    grad_tangent = grad_euclidean - dots[:, np.newaxis] * points
    
    return grad_tangent

def optimize_riemannian(points, max_iter=2000, init_lr=0.01, decay=0.999):
    """
    Perform Riemannian Gradient Descent with momentum.
    """
    x = points.copy()
    momentum = np.zeros_like(x)
    mu = 0.9 # Momentum coefficient
    lr = init_lr
    
    for i in range(max_iter):
        grad = compute_riemannian_gradient(x)
        
        # Update momentum
        momentum = mu * momentum + lr * grad
        
        # Update points
        x_new = x - momentum
        
        # Normalize back to sphere
        norms = np.linalg.norm(x_new, axis=1, keepdims=True)
        # Handle zero norms (should not happen but safe)
        norms = np.maximum(norms, 1e-15)
        x = x_new / norms
        
        # Decay learning rate
        lr *= decay
        
        # Optional: Check convergence
        # if np.linalg.norm(momentum) < 1e-8:
        #     break
        
    return x

def construct():
    best_energy = np.inf
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    inits = []
    
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(init_fib)
    
    # 2. Random Points
    for _ in range(5):
        v = rng.normal(size=(N, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        inits.append(v)
        
    # 3. Jittered Fibonacci (adds noise to break symmetries)
    for _ in range(10):
        noise = rng.normal(scale=0.1, size=(N, 3))
        v = init_fib + noise
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        inits.append(v)
        
    # 4. Clustered Random (sometimes helps escape bad local mins if optimized well, 
    #    but usually bad. Let's try some points starting near poles)
    for _ in range(2):
        # Start with two clusters at poles, then let them repel
        v = np.zeros((N, 3))
        half = N // 2
        v[:half] = rng.normal(size=(half, 3))
        v[half:] = rng.normal(size=(N-half, 3))
        v[:half, 2] += 10.0
        v[half:, 2] -= 10.0
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        inits.append(v)

    for init in inits:
        # Run Riemannian Gradient Descent
        # Use a few iterations with high LR to spread points, then lower LR to refine
        # Simple decay is usually sufficient
        
        # Phase 1: Coarse optimization
        res1 = optimize_riemannian(init, max_iter=1000, init_lr=0.05, decay=0.998)
        
        # Phase 2: Fine optimization
        res2 = optimize_riemannian(res1, max_iter=2000, init_lr=0.01, decay=0.999)
        
        # Calculate final energy
        energy = compute_energy(res2)
        
        if energy < best_energy:
            best_energy = energy
            best_points = res2.copy()
            
    # Final cleanup: ensure no points are too close
    # If points are closer than 1e-9, we might have a numerical issue.
    # The optimizer should prevent this, but let's verify.
    
    return best_points
