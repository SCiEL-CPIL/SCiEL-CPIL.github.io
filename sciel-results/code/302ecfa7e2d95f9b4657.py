import numpy as np
import math

N = 150

def compute_energy(points):
    """
    Compute the logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)
    """
    # Vectorized distance calculation
    # diff shape (N, N, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # We only need i < j
    triu = np.triu_indices(N, k=1)
    d_pair = np.sqrt(dist_sq[triu])
    
    # Avoid log(0)
    d_pair = np.maximum(d_pair, 1e-15)
    
    energy = -np.sum(np.log(d_pair))
    return energy

def compute_gradient(points):
    """
    Compute the Euclidean gradient of the logarithmic energy.
    E = sum_{i<j} -log(||x_i - x_j||)
    dE/dx_i = sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    Note: The term for j=i is 0.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # inv_dist_sq shape (N, N)
    inv_dist_sq = 1.0 / dist_sq
    
    # Gradient for each point i is - sum_j (x_i - x_j) / ||x_i - x_j||^2 ?
    # Let's check signs.
    # E = - sum log(d). dE/dd = -1/d.
    # dd/dx_i = (x_i - x_j)/d.
    # dE/dx_i = sum_{j} (-1/d) * (x_i - x_j)/d = - sum (x_i - x_j) / d^2.
    # So grad_i = - sum_j (x_i - x_j) / ||x_i - x_j||^2
    
    # diff[i, j] = x_i - x_j
    # We want sum over j for each i.
    # grad = - sum_j (diff[i, j] * inv_dist_sq[i, j])
    
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return grad

def riemannian_gradient_descent(points, steps=2000, base_lr=0.01, annealing_factor=0.999, noise_scale=0.01):
    """
    Perform Riemannian Gradient Descent with annealing.
    """
    p = points.copy()
    current_energy = compute_energy(p)
    best_energy = current_energy
    best_p = p.copy()
    
    lr = base_lr
    noise = noise_scale
    
    for step in range(steps):
        # Compute Euclidean gradient
        grad_eucl = compute_gradient(p)
        
        # Project to tangent space
        # g_manifold = grad_eucl - (grad_eucl . p) * p
        # dot product of grad and p for each point
        dot = np.sum(grad_eucl * p, axis=1, keepdims=True)
        grad_manifold = grad_eucl - dot * p
        
        # Add random noise to escape local minima
        # Generate random vectors and project them to tangent space too? 
        # Simpler: add small random perturbation to p directly, or to gradient.
        # Let's add noise to the gradient direction.
        rng = np.random.default_rng(step)
        noise_vec = rng.normal(size=p.shape)
        # Project noise to tangent space to ensure we stay on sphere manifold direction
        noise_dot = np.sum(noise_vec * p, axis=1, keepdims=True)
        noise_tangent = noise_vec - noise_dot * p
        
        grad_total = grad_manifold + noise * noise_tangent
        
        # Update
        p = p - lr * grad_total
        
        # Normalize back to sphere
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        p = p / norms
        
        # Compute energy
        current_energy = compute_energy(p)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_p = p.copy()
        
        # Annealing
        lr *= annealing_factor
        noise *= annealing_factor
        
    return best_p, best_energy

def construct():
    best_energy = np.inf
    best_points = None
    
    # Multi-start strategy
    # 1. Fibonacci sphere (good uniformity)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Random initializations
    rng = np.random.default_rng(42)
    
    inits = [init_fib]
    for _ in range(15):
        v = rng.normal(size=(N, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        inits.append(v)
        
    # Run Riemannian GD for each initialization
    for init in inits:
        # Use a fixed number of steps, tuned for convergence
        # We can increase steps if we have time, but 120s limit is tight.
        # 150 points, 15 starts, 2000 steps = 45000 iterations.
        # Each iteration involves O(N^2) operations. N=150 -> 22500 ops.
        # 45000 * 22500 = ~1e9 ops. Might be slow in pure Python/NumPy loops if not optimized.
        # Let's reduce steps or parallelize? No parallel.
        # Let's try 500 steps first, then refine best ones.
        
        p_opt, e_opt = riemannian_gradient_descent(init, steps=500, base_lr=0.05, annealing_factor=0.998, noise_scale=0.05)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_points = p_opt.copy()
            
    # Refine the best point with more steps and lower noise
    if best_points is not None:
        p_final, e_final = riemannian_gradient_descent(best_points, steps=2000, base_lr=0.01, annealing_factor=0.9995, noise_scale=0.001)
        if e_final < best_energy:
            best_energy = e_final
            best_points = p_final
            
    return best_points
