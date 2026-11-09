import numpy as np
from math import fsum

N = 150

# EVOLVE-BLOCK-START

def _fibonacci_spiral(n, seed_shift=0.0):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5 + seed_shift
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_energy(x):
    """Compute logarithmic energy sum_{i<j} log(1/d_ij)."""
    n = len(x)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    # Only sum upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    log_terms = np.log(1.0 / dist) * mask
    return fsum(log_terms[mask].tolist())

def _gradient_descent(x, num_steps=8000, initial_lr=0.08):
    """Run gradient descent on sphere to minimize log energy."""
    x = x.copy()
    n = len(x)
    
    for step in range(num_steps):
        # Vectorized distance calculation
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        
        inv_dist_sq = 1.0 / dist_sq
        inv_dist_sq[~np.isfinite(inv_dist_sq)] = 0.0
        
        force = diff * inv_dist_sq[:, :, np.newaxis]
        total_force = np.sum(force, axis=1)
        
        # Project onto tangent plane
        dot_products = np.sum(total_force * x, axis=1)
        tangent_force = total_force - dot_products[:, np.newaxis] * x
        
        # Adaptive learning rate with cosine-like decay
        progress = step / num_steps
        lr = initial_lr * (0.5 * (1 + np.cos(np.pi * progress)))
        
        x = x + lr * tangent_force
        
        # Normalize to unit sphere
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        x = x / norms
    
    return x

def construct():
    # Try multiple initializations and keep the best
    best_x = None
    best_energy = np.inf
    
    # Initialization strategies
    initializations = []
    
    # 1. Standard Fibonacci spiral
    initializations.append(_fibonacci_spiral(N))
    
    # 2. Fibonacci with slight shifts
    for shift in [0.1, 0.2, 0.3]:
        initializations.append(_fibonacci_spiral(N, seed_shift=shift))
    
    # 3. Random initialization with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    for _ in range(3):
        rand = rng.randn(N, 3)
        rand /= np.linalg.norm(rand, axis=1, keepdims=True)
        initializations.append(rand)
    
    # 4. Symmetric initialization: place points in rings
    # 5 rings of 30 points each at different latitudes
    ring_lats = np.linspace(-0.8, 0.8, 5)
    ring_points = []
    for lat in ring_lats:
        z = np.sin(lat)
        r = np.cos(lat)
        angles = np.linspace(0, 2*np.pi, 30, endpoint=False)
        ring = np.column_stack([r * np.cos(angles), r * np.sin(angles), np.full(30, z)])
        ring_points.append(ring)
    initializations.append(np.vstack(ring_points))
    
    # Run gradient descent from each initialization
    for init in initializations:
        if not np.all(np.isfinite(init)):
            continue
        
        # Quick check: minimum distance
        diff = init[:, np.newaxis, :] - init[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        if dist.min() < 1e-6:
            continue
        
        # Run optimization
        optimized = _gradient_descent(init, num_steps=6000, initial_lr=0.08)
        
        # Validate
        if not np.all(np.isfinite(optimized)):
            continue
        
        # Check minimum distance
        diff = optimized[:, np.newaxis, :] - optimized[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        if dist.min() < 1e-9:
            continue
        
        # Compute energy
        energy = _compute_energy(optimized)
        
        if energy < best_energy:
            best_energy = energy
            best_x = optimized
    
    # If all failed, fallback to Fibonacci
    if best_x is None or not np.all(np.isfinite(best_x)):
        best_x = _fibonacci_spiral(N)
    
    return best_x

# EVOLVE-BLOCK-END
