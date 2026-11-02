import numpy as np
import time

N = 50
DEADLINE = 145.0  # Leave 5 seconds margin

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _covering_radius(X, samples):
    # X: (N, 3)
    # samples: (M, 3)
    # Compute distance from each sample to each point in X
    # dist: (M, N)
    # We want max over M of min over N of dist
    
    # Use dot product to compute angular distance
    # cos(theta) = samples @ X.T
    cos_theta = samples @ X.T
    # Clip to avoid numerical issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    min_theta = np.min(theta, axis=1)
    return np.max(min_theta)

def construct():
    start_time = time.time()
    
    # Initial configuration
    X = _fibonacci_sphere(N)
    
    # Sample points for covering radius estimation
    M = 5000
    samples = _fibonacci_sphere(M)
    
    best_X = X.copy()
    best_radius = _covering_radius(X, samples)
    
    # Simulated Annealing
    T = 0.1  # Initial temperature
    cooling_rate = 0.999
    step_size = 0.1
    
    while time.time() - start_time < DEADLINE:
        # Perturb a random point
        idx = np.random.randint(N)
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        
        new_X = X.copy()
        new_X[idx] = X[idx] + step_size * direction
        new_X[idx] /= np.linalg.norm(new_X[idx])
        
        new_radius = _covering_radius(new_X, samples)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_X = new_X.copy()
        
        # Metropolis criterion
        delta = new_radius - _covering_radius(X, samples)
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            X = new_X
        
        T *= cooling_rate
        if T < 1e-6:
            break
            
    return best_X
