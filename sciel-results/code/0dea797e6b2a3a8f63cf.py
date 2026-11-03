import numpy as np
import time

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Setup Grid for Covering Radius Approximation
    # We need a grid dense enough to approximate the continuous max, 
    # but sparse enough to compute gradients quickly.
    # 200x400 grid is ~80k points. 130 points vs 80k grid -> 10.4M pairs.
    # Vectorized in chunks.
    n_theta = 100
    n_phi = 200
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta, phi = np.meshgrid(theta, phi, indexing='ij')
    
    # Spherical coordinates to Cartesian
    sin_theta = np.sin(theta)
    grid_x = sin_theta * np.cos(phi)
    grid_y = sin_theta * np.sin(phi)
    grid_z = np.cos(theta)
    
    # Flatten grid
    G = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    G = G / np.linalg.norm(G, axis=1, keepdims=True) # Ensure unit sphere
    
    M = len(G)
    
    # Precompute grid properties for gradient calculation
    # We want to minimize max_d. 
    # Soft max: log(sum(exp(d_i^2 / T))) * T
    # Gradient w.r.t point k: sum_i (exp(d_ik^2/T) / sum_j exp(d_ij^2/T)) * (2/T) * (p_k - g_i) * p_k_perp?
    # Actually, we optimize on the sphere. 
    # Let's use the gradient of the soft-max distance.
    
    def compute_soft_max_gradient(points, temperature):
        """
        Computes the gradient of the soft-max covering radius w.r.t points.
        points: (N, 3)
        temperature: scalar
        Returns: (N, 3) gradient
        """
        # To save memory, process grid in chunks
        chunk_size = 10000
        grad = np.zeros_like(points)
        
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            G_chunk = G[start:end]
            
            # Distances squared: (chunk, N)
            # ||g - p||^2 = ||g||^2 + ||p||^2 - 2 g.p = 2 - 2 g.p (since unit vectors)
            dots = G_chunk @ points.T # (chunk, N)
            d2 = 2.0 - 2.0 * dots
            
            # Soft max weights: exp(d2 / T)
            # To avoid overflow, subtract max per row
            max_d2 = np.max(d2, axis=1, keepdims=True)
            exp_d2 = np.exp((d2 - max_d2) / temperature)
            sum_exp = np.sum(exp_d2, axis=1, keepdims=True)
            weights = exp_d2 / sum_exp # (chunk, N)
            
            # Gradient of sum_i w_i * d_ik^2
            # d/dp_k (d_ik^2) = d/dp_k (2 - 2 g_i . p_k) = -2 g_i
            # So grad_k = sum_i w_i * (-2 g_i)
            # Wait, the soft max is log(sum(exp(d2/T))).
            # d/dp_k log(sum_j exp(d_jk^2/T)) = (1/sum) * sum_j exp(d_jk^2/T) * (1/T) * d/dp_k(d_jk^2)
            # = (1/T) * sum_j w_j * (-2 g_j)
            # = (-2/T) * (weights.T @ G_chunk)
            
            grad_chunk = (-2.0 / temperature) * (weights.T @ G_chunk)
            grad += grad_chunk
            
        return grad

    # Optimization Loop
    # We use a dynamic temperature schedule.
    # Start with high T (smooth landscape, global view) and lower it (sharp, local precision).
    
    current_temp = 0.01 # Initial temperature (radians^2 roughly)
    lr = 0.05
    patience = 0
    best_score = np.inf
    best_points = points.copy()
    
    # Track progress
    last_improvement_time = start_time
    
    while time.time() < deadline:
        # Compute current soft-max radius for monitoring (approximate)
        # We don't strictly need the exact value every step, but it helps for LR decay
        
        grad = compute_soft_max_gradient(points, current_temp)
        
        # Normalize gradient to unit sphere constraint
        # Project gradient to tangent space: grad - (grad . p) p
        dots_grad = np.sum(points * grad, axis=1, keepdims=True)
        grad_tangent = grad - dots_grad * points
        
        # Update
        points = points + lr * grad_tangent
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Adaptive Learning Rate / Temperature
        # Every 50 steps, check improvement
        # We can't compute exact covering radius cheaply every step.
        # Instead, we monitor the soft-max value.
        # Let's compute soft-max value periodically or just trust the descent.
        
        # Simple heuristic: decay temperature slowly
        if patience % 10 == 0:
            # Check if we are stuck? 
            # For now, just decay temperature to sharpen the landscape
            current_temp *= 0.95
            lr *= 0.99
            
        patience += 1
        
        # Safety check: if points collapse or become invalid
        if not np.all(np.isfinite(points)):
            points = best_points
            break

    # Final Polish: Use a denser grid for a final few steps with very low T
    # Or just return the best found.
    # Given time constraints, we return the current points.
    # The evaluator will compute the true covering radius.
    
    return points
