import numpy as np
import time

N = 50

def _sample_covering_radius(x, n_samples=5000):
    """Estimate covering radius by sampling points on the sphere."""
    # Generate random points on sphere
    # Uniform distribution
    u = np.random.rand(n_samples)
    v = np.random.rand(n_samples)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    
    y = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    
    # Calculate min distance from each y to any x
    # y: (M, 3), x: (N, 3)
    # dist: (M, N)
    # dist_sq = ||y||^2 + ||x||^2 - 2 y.x = 1 + 1 - 2 y.x = 2 - 2 y.x
    # Since we want angular distance, cos(d) = y.x
    # d = arccos(y.x)
    # We want max d.
    
    dots = y @ x.T  # (M, N)
    # Clip to [-1, 1] for arccos stability
    dots = np.clip(dots, -1.0, 1.0)
    
    min_dots = np.min(dots, axis=1) # Max distance corresponds to Min dot product
    max_d = np.arccos(min_dots)
    
    return np.max(max_d)

def _optimize_repulsion(x0, p=20, steps=3000, eta=0.01):
    x = x0.copy()
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    start_time = time.time()
    time_limit = 10.0 # 10s per run
    
    for step in range(steps):
        if time.time() - start_time > time_limit:
            break
            
        diff = x[i_idx] - x[j_idx]
        d_sq = np.sum(diff**2, axis=1)
        d = np.sqrt(d_sq + 1e-12)
        
        factor = -p * (d ** (-p - 2))
        
        grad = np.zeros_like(x)
        np.add.at(grad, i_idx, factor[:, None] * diff)
        np.add.at(grad, j_idx, -factor[:, None] * diff)
        
        # Decay step size
        current_eta = eta * (0.999 ** step)
        
        x_new = x - current_eta * grad
        norms = np.linalg.norm(x_new, axis=1, keepdims=True)
        x_new = x_new / (norms + 1e-12)
        
        x = x_new
        
    return x

def construct():
    np.random.seed(42)
    
    best_x = None
    best_score = np.inf
    
    # Try a few initializations
    for run in range(5):
        # Fibonacci spiral with slight noise
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Add noise
        if run > 0:
            noise = np.random.randn(N, 3) * 0.1
            x = x + noise
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            
        x_opt = _optimize_repulsion(x, p=20, steps=2000, eta=0.01)
        
        score = _sample_covering_radius(x_opt, n_samples=10000)
        
        if score < best_score:
            best_score = score
            best_x = x_opt
            
        # Check time
        if time.time() > 100:
            break
            
    return best_x
