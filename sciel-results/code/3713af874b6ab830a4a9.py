import time
import numpy as np

def priority(adjacency):
    """
    Score vertices using a fixed-point repulsion heuristic for Maximum Independent Set.
    
    Strategy:
    1. Base score is inversely proportional to degree (low degree is good).
    2. Iteratively penalize scores based on the sum of neighbor scores.
       This models the greedy blocking effect: if a vertex has neighbors with high potential,
       its own potential is reduced because it's likely to be blocked or block them.
    3. The fixed-point iteration converges to a stable assignment where scores reflect
       the local competitive landscape.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit

    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]

    if n == 0:
        return np.array([], dtype=np.float64)
    
    if n == 1:
        return np.array([0.0], dtype=np.float64)

    # Ensure symmetry
    A = (A + A.T) / 2.0

    # Compute degree vector
    deg = A.sum(axis=1)
    
    # Base score: Inverse degree
    # Low degree vertices are preferred candidates for MIS
    S_base = 1.0 / (deg + 1.0)
    
    # Normalization is not strictly necessary for sorting, but helps with scaling
    # in the fixed-point equation.
    
    # Fixed-point iteration parameters
    # lambda controls the strength of the "repulsion" between neighbors.
    # Higher lambda means stronger penalty for having high-scoring neighbors.
    lam = 1.5
    alpha = 0.8 # Damping for stability
    
    # Initialize scores
    S = S_base.copy()
    
    # Run fixed-point iterations
    max_iters = 100
    for k in range(max_iters):
        if time.time() > DEADLINE:
            break
            
        # Compute sum of neighbor scores
        neighbor_sum = A @ S
        
        # Update scores
        # S_new = S_base / (1 + lambda * neighbor_sum)
        # This ensures that if neighbors have high scores, S drops.
        # If neighbors have low scores, S stays close to S_base.
        S_new = S_base / (1.0 + lam * neighbor_sum)
        
        # Damped update to prevent oscillation
        S = (1.0 - alpha) * S + alpha * S_new
        
        # Optional: Check for convergence
        # diff = np.abs(S_new - S).max()
        # if diff < 1e-6:
        #     break

    # Final normalization to [0, 1] for consistency (optional, but good practice)
    s_min = S.min()
    s_max = S.max()
    if s_max > s_min:
        S = (S - s_min) / (s_max - s_min)
    else:
        S = np.zeros(n)

    return S.astype(np.float64)
