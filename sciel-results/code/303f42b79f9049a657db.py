import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = 110.0  # Safety margin before 150s kill
    
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    # Calculate degrees
    degrees = A.sum(axis=1)
    
    # Calculate local edge density (clustering-like metric)
    # B = A^2, B[v,u] = number of common neighbors of v and u
    # two_L[v] = sum_{u in N(v)} B[v,u] = 2 * L(v) where L(v) = edges in N(v)
    B = A @ A
    two_L = np.sum(A * B, axis=1)
    L = two_L / 2.0
    
    # Local edge density: rho(v) = 2 * L(v) / (d(v) * (d(v) - 1))
    denom = degrees * (degrees - 1.0)
    denom[denom < 1e-10] = 1.0
    rho = np.where(degrees >= 2, two_L / denom, 0.0)
    
    # Iterative refinement of vertex weights
    # Base weight: low degree is more valuable (easier to keep in independent set)
    # Also incorporate local density: high density neighbors are less "costly" to lose
    # because they conflict with each other
    base_weights = 1.0 / (degrees + 1.0)
    
    # Initialize weights with a blend of inverse degree and density bonus
    # Vertices with low degree AND low local density are most valuable
    weights = base_weights * (1.0 - 0.3 * rho)
    weights = np.maximum(weights, 1e-6)  # Avoid zero weights
    
    # Iterative refinement: update weights based on neighbor structure
    # A vertex's value is reduced if its neighbors are also high-value
    # (they compete for selection, so picking one loses the other)
    for iteration in range(15):
        if time.time() - start_time > deadline:
            break
        
        # Sum of neighbor weights
        neighbor_weight_sum = A @ weights
        
        # Updated weight: base value adjusted for neighbor competition
        # Higher neighbor weight sum means more competition -> lower effective value
        # Also incorporate local density: high density means neighbors conflict,
        # so losing them is less costly
        weights_new = base_weights * (1.0 - 0.3 * rho) / (1.0 + 0.5 * neighbor_weight_sum)
        
        # Check for convergence
        if np.allclose(weights, weights_new, atol=1e-10):
            weights = weights_new
            break
        weights = weights_new
    
    # Final score: negative cost of picking each vertex
    # Cost components:
    # 1. Degree (number of edges removed)
    # 2. Sum of neighbor weights (value of eliminated vertices)
    # 3. Density bonus: high local density means neighbors conflict, reducing effective cost
    neighbor_value_sum = A @ weights
    
    # Score = - (degree + neighbor_value_sum) + density_bonus
    # The density bonus rewards picking vertices whose neighbors are tightly connected
    # (because those neighbors would conflict anyway, so losing them is less costly)
    scores = -(degrees + neighbor_value_sum) + 0.5 * two_L
    
    # Safety check
    if time.time() - start_time > deadline:
        return scores
    
    return scores
