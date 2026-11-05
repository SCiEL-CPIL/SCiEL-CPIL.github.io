import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    Use spectral centrality (principal eigenvector) to identify peripheral nodes.
    Vertices with low eigenvector centrality are prioritized (higher score) as they 
    are less entangled in the core of the graph. This is a global structural measure 
    differing from local degree-based heuristics.
    
    We compute the principal eigenvector via power iteration.
    Score = -alpha * eigenvector_component + beta * (1 / (degree + 1))
    
    Rationale:
    - Low eigenvector centrality -> High score -> Selected earlier.
    - These nodes are often on the "fringes" or in less dense parts of the graph,
      making them safer picks that don't block as many potential independent set members
      in the core.
    """
    start_time = time.time()
    
    # Convert to float array for fast vectorized operations
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # 1. Calculate degrees for the secondary term
    deg = adj.sum(axis=1)
    
    # 2. Compute Principal Eigenvector via Power Iteration
    # Initialize with a random or uniform vector. Uniform is deterministic and safe.
    v = np.ones(n)
    v = v / np.linalg.norm(v)
    
    # Power iteration steps. 
    # For n=200, convergence is fast. 100 iterations is very safe and fast.
    # We check time every 10 iterations to be safe.
    max_iters = 100
    for i in range(max_iters):
        # v_new = A @ v
        v_new = adj @ v
        norm = np.linalg.norm(v_new)
        if norm == 0:
            # If the graph is empty or disconnected in a way that zeroes out, break
            break
        v_new = v_new / norm
        
        # Check convergence (optional, but helps stop early if needed)
        # diff = np.linalg.norm(v_new - v)
        # if diff < 1e-6:
        #     break
        
        v = v_new
        
        # Time check
        if i % 10 == 0 and time.time() - start_time > 100:
            break
            
    # 3. Combine Scores
    # We want to SELECT vertices with LOW eigenvector centrality.
    # So the score should be negatively correlated with the eigenvector value.
    # score = -eigenvector_value
    
    # Normalize eigenvector to [0, 1] range for better scaling?
    # Or just use raw values. Power iteration gives unit norm.
    # Values can be negative. We should probably take the absolute value or just use the value.
    # Since A is non-negative, the principal eigenvector can be taken non-negative (Perron-Frobenius).
    # However, if the graph is disconnected, components might have different signs or zeros.
    # To be robust, let's use the magnitude or just the value assuming non-negative initialization leads to non-negative vector.
    # With v0 = ones, and A >= 0, v remains >= 0.
    
    eig_vec = v
    
    # Secondary term: prefer low degree as well, but less strongly?
    # The prompt suggests spectral centrality is the main driver.
    # Let's try: Score = -eig_vec + 0.1 * (1/(deg+1))
    # The degree term helps break ties or handle isolated nodes (deg=0, eig=0 -> score 1.0).
    # For a node in the core, eig is high, score is very negative.
    # For a leaf, eig is low, score is higher.
    
    # Let's ensure the scale is comparable.
    # Eigenvector values are typically small (1/sqrt(n) ~ 0.07 for n=200).
    # 1/(deg+1) is in [0, 1].
    # So the degree term will dominate if not scaled.
    # Let's scale eigenvector to max 1.
    max_eig = np.max(eig_vec)
    if max_eig > 0:
        eig_norm = eig_vec / max_eig
    else:
        eig_norm = eig_vec
        
    # Score: Higher is better.
    # We want LOW eig_norm to have HIGH score.
    # So score = -eig_norm.
    
    # Add a small degree penalty to prefer lower degree if eigenvectors are similar.
    # Or maybe just pure eigenvector?
    # Let's try a weighted sum.
    # Weight 1.0 for eig, 0.5 for inverse degree?
    
    inv_deg = 1.0 / (deg + 1.0)
    
    # Normalize inv_deg to [0, 1] range? Max is 1. Min is ~0.
    
    score = -1.0 * eig_norm + 0.5 * inv_deg
    
    # Safety deadline check
    if time.time() - start_time > 140:
        # Fallback to simple LDF
        return -deg
        
    return score
