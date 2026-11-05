"""Program using Katz centrality for Maximum Independent Set priority scoring."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign scores to vertices based on Katz centrality.
    Vertices with lower Katz centrality (less central, lower degree/connectivity) 
    get higher priority (higher score in descending order).
    
    Katz centrality is defined as x = (I - alpha * A)^(-1) * 1.
    We compute it iteratively: x_{k+1} = 1 + alpha * A * x_k.
    We prioritize vertices with low Katz centrality, which typically corresponds 
    to vertices with low degree and low connectivity to high-degree hubs.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    n = adjacency.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Estimate spectral radius (lambda_max) using power iteration
    # This helps in choosing a safe alpha for Katz iteration
    if np.all(adjacency == 0):
        # No edges, all scores equal
        return np.zeros(n)
        
    # Simple power iteration for spectral radius
    rng = np.random.RandomState(42)
    v = rng.rand(n)
    v = v / np.linalg.norm(v)
    lambda_max = 0.0
    for _ in range(100):
        v_new = adjacency @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-12:
            break
        lambda_max = norm
        v = v_new / norm
        
    if lambda_max < 1e-12:
        lambda_max = 1.0
        
    # Choose alpha such that alpha * lambda_max < 1 for convergence
    # 0.5 is a safe factor
    alpha = 0.5 / lambda_max
    
    # Compute Katz Centrality
    # x = 1 + alpha * A * x
    # Initialize x with 1s
    x = np.ones(n)
    
    for _ in range(200):
        x_new = 1.0 + alpha * (adjacency @ x)
        # Check convergence
        if np.linalg.norm(x_new - x) < 1e-10:
            x = x_new
            break
        x = x_new
        
    # Score: We want to pick vertices with LOW centrality first.
    # So, score = -centrality.
    scores = -x
    
    return scores
# EVOLVE-BLOCK-END
