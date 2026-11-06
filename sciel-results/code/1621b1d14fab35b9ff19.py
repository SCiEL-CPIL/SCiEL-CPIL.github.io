"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign a priority score to each vertex for greedy independent set selection.
    
    Approach:
    1. Formulate the problem using the spectral properties of the complement graph.
    2. The complement graph's adjacency matrix is B = J - I - A.
    3. The leading eigenvector of B favors vertices that are "well-connected" in the 
       complement graph, which corresponds to vertices with low degree in the original 
       graph and, more importantly, vertices in sparse regions of the original graph.
    4. We use power iteration to find the leading eigenvector of B efficiently, 
       avoiding explicit construction of the all-ones matrix J.
       The update rule is: v_new = (sum(v)) * 1 - v - A @ v
    5. Combine this global spectral score with a local degree-based score to refine 
       the priorities.
    """
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Degrees for local heuristic
    degrees = adj.sum(axis=1)
    
    # Power iteration for the leading eigenvector of the complement graph (excluding self-loops)
    # Matrix B = J - I - A
    # B @ v = J @ v - I @ v - A @ v
    # J @ v = (sum(v)) * ones
    
    # Initialize v with small random noise to avoid symmetry breaking issues.
    # A fixed seed ensures deterministic behavior for a given graph.
    np.random.seed(42)
    v = np.random.uniform(0.1, 1.0, n)
    
    max_iter = 100
    for _ in range(max_iter):
        # Compute A @ v
        av = adj @ v
        
        # Compute sum(v)
        s = np.sum(v)
        
        # Update: v_new = s * 1 - v - A @ v
        v_new = s * np.ones(n) - v - av
        
        # Normalize to prevent overflow
        norm = np.linalg.norm(v_new)
        if norm < 1e-12:
            break
        v_new = v_new / norm
        
        # Check convergence
        if np.linalg.norm(v_new - v) < 1e-7:
            v = v_new
            break
        v = v_new
        
    # The leading eigenvector might have negative components.
    # For ranking, we care about the relative magnitude.
    # Shift v to be non-negative.
    v = v - np.min(v)
    
    # Local degree score: lower degree is better for independent set.
    
    def normalize(v):
        v_min, v_max = v.min(), v.max()
        if v_max - v_min < 1e-12:
            return np.zeros_like(v)
        return (v - v_min) / (v_max - v_min)
    
    spectral_score = normalize(v)
    degree_score = normalize(-degrees)
    
    # Combine: The spectral score of the complement graph is a strong indicator 
    # of "independence potential". 
    # Vertices with high spectral score are in sparse areas.
    combined = 0.8 * spectral_score + 0.2 * degree_score
    
    return combined
# EVOLVE-BLOCK-END
