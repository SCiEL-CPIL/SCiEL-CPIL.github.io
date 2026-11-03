"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math


def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = points.shape[0]
    # Compute all pairwise distances squared
    # dist_sq = 2 - 2 * (points @ points.T)
    G = points @ points.T
    dist_sq = 2.0 - 2.0 * G
    np.fill_diagonal(dist_sq, 1.0)
    
    # Extract upper triangle
    iu = np.triu_indices(n, k=1)
    d_sq_upper = dist_sq[iu]
    
    # Clip to avoid log(0) or log(negative due to numerical error)
    d_sq_upper = np.maximum(d_sq_upper, 1e-30)
    
    # Energy is sum log(1/sqrt(d_sq)) = -0.5 sum log(d_sq)
    terms = -0.5 * np.log(d_sq_upper)
    return math.fsum(terms)


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy using Adam on the Sphere."""
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    x = x / np.linalg.norm(x, axis=1, keepdims=True)

    # Adam Parameters
    alpha = 0.05  # Learning rate
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    max_iter = 5000
    
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    best_energy = _log_energy(x)
    best_x = x.copy()
    
    # We will run Adam for max_iter steps
    for step in range(max_iter):
        t += 1
        
        # Compute Gradients
        # g_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
        # Let D be matrix of distances squared.
        G = x @ x.T
        dist_sq = 2.0 - 2.0 * G
        np.fill_diagonal(dist_sq, 0.0) # Diagonal is 0, will be masked
        
        # We want sum over j of (x_i - x_j) / dist_sq_ij
        # (x_i - x_j) is not a simple matrix op directly for all i,j simultaneously in a vectorized way without broadcasting carefully.
        # x_i - x_j can be written as x_i * 1 - 1 * x_j.
        # Let's use broadcasting: x[:, np.newaxis, :] - x[np.newaxis, :, :]
        # This creates a (N, N, 3) array. For N=100, this is 10,000 * 3 floats, which is fine.
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        # dist_sq is (N, N)
        # We need to divide diff by dist_sq[..., np.newaxis]
        # But dist_sq diagonal is 0. We should mask it.
        
        # Mask diagonal to avoid division by zero
        mask = np.ones((N, N), dtype=bool)
        np.fill_diagonal(mask, False)
        
        # For numerical stability, replace 0 distances in off-diagonals (if any) with small value?
        # No, if points are close, force is large.
        # We can add a small epsilon to dist_sq to prevent inf, but that changes the physics slightly.
        # Given N=100 and random/Fib start, collisions are unlikely to be exact zeros.
        
        safe_dist_sq = np.where(mask, dist_sq, 1.0)
        safe_dist_sq = np.maximum(safe_dist_sq, 1e-15)
        
        # Force term: (x_i - x_j) / dist_sq_ij
        # Shape: (N, N, 3) / (N, N, 1)
        force_terms = diff / safe_dist_sq[..., np.newaxis]
        
        # Sum over j (axis 1)
        # The gradient of Energy is g_i = - sum_{j} force_terms[i,j]
        # So we want to move in direction -g_i = sum force_terms
        grad = -np.sum(force_terms, axis=1)
        
        # Update Adam states
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Update x
        x = x - alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Normalize to sphere
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / norms
        
        # Check energy
        if step % 10 == 0:
            current_energy = _log_energy(x)
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()
                
    return best_x
# EVOLVE-BLOCK-END
