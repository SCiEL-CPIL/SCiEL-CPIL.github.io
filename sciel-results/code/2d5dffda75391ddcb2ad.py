import numpy as np
import time

N = 150

def _fibonacci_lattice(N):
    """Generate Fibonacci lattice points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])

def _energy_and_grad(x):
    """
    Calculate logarithmic energy and gradient.
    x: (N, 3) array of points.
    Returns: energy (float), grad (N, 3) array.
    Energy = sum_{i<j} log(1 / ||x_i - x_j||)
    """
    # Compute pairwise differences
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    # Squared distances
    d2 = np.sum(diff * diff, axis=2)
    # Avoid division by zero (diagonal)
    np.fill_diagonal(d2, 1.0)
    
    # Distances
    d = np.sqrt(d2)
    
    # Energy: sum_{i<j} log(1/d_ij)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    e_vals = np.log(1.0 / d[mask])
    energy = np.sum(e_vals)
    
    # Gradient
    # grad_i = - sum_{j != i} (x_i - x_j) / d_ij^2
    inv_d2 = 1.0 / d2
    terms = diff * inv_d2[:, :, np.newaxis]
    forces = np.sum(terms, axis=1)
    grad = -forces
    
    return energy, grad

def construct():
    start_time = time.time()
    # 145s limit to be safe
    deadline = start_time + 145.0
    
    # Initial points
    x = _fibonacci_lattice(N)
    
    # Add small random noise to break symmetry
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01, x.shape)
    x = x + noise
    # Normalize
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / norms
    
    current_energy, current_grad = _energy_and_grad(x)
    best_energy = current_energy
    best_x = x.copy()
    
    # Gradient Descent with Line Search
    step_size = 0.01
    
    while time.time() < deadline:
        # Project gradient onto tangent space
        # grad_proj = grad - (grad . x) x
        # This ensures the step is tangent to the sphere
        # (x . grad) is computed for each point
        grad_dot_x = np.sum(x * current_grad, axis=1, keepdims=True)
        grad_proj = current_grad - grad_dot_x * x
        
        # Line search
        alpha = step_size
        improved = False
        
        for _ in range(20):
            x_proposed = x - alpha * grad_proj
            # Normalize
            x_proposed = x_proposed / np.linalg.norm(x_proposed, axis=1, keepdims=True)
            
            e_proposed, g_proposed = _energy_and_grad(x_proposed)
            
            if e_proposed < current_energy:
                x = x_proposed
                current_energy = e_proposed
                current_grad = g_proposed
                improved = True
                break
            else:
                alpha *= 0.5
        
        if not improved:
            if alpha < 1e-12:
                break
            # If no improvement, break to avoid infinite loop on numerical noise
            break
            
        if current_energy < best_energy:
            best_energy = current_energy
            best_x = x.copy()
            
        # Optional: adapt step size
        # If we improved with full step, maybe increase slightly?
        # If we had to backtrack a lot, decrease?
        # Keep simple for now.
        
    return best_x
