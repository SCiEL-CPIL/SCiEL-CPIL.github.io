"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign a priority score to each vertex for greedy independent set selection.
    
    Approach:
    1. Convert adjacency to a numpy array.
    2. Compute degrees.
    3. Use a spectral relaxation approach: 
       - Construct a weighted adjacency matrix where edge weights represent
         the marginal loss of selecting both endpoints (proportional to degrees).
       - Use power iteration on a modified Laplacian to find a score vector
         that captures global structural constraints.
    4. Combine spectral scores with local degree-based heuristics.
    """
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # Degrees
    degrees = adj.sum(axis=1)
    
    # Handle empty graph
    if n == 0:
        return np.array([], dtype=float)
    
    # Spectral approach:
    # We want to find a vector x such that vertices with high x are "good" to pick.
    # The idea: in a Max Weight Independent Set formulation, we can use a relaxation
    # where the objective is sum_i w_i x_i subject to x_i + x_j <= 1 for edges (i,j).
    # 
    # A spectral relaxation: consider the matrix A = adjacency.
    # The leading eigenvector of (I - A/n) or similar can give global structure.
    # 
    # Alternative: use the fact that in a graph, vertices with low "connectivity"
    # (low degree, in low-clustering regions) are better to pick.
    #
    # Let's use a power iteration on a matrix that encodes "independence potential":
    # M = I - (1/n) * A  (normalized adjacency)
    # The leading eigenvector of M gives a score where vertices that are
    # "far" from dense regions get higher scores.
    
    # Normalize adjacency for stability
    degree_sum = degrees.sum()
    if degree_sum > 0:
        # Normalize by average degree for scale invariance
        avg_deg = degree_sum / n
        if avg_deg > 0:
            A_norm = adj / avg_deg
        else:
            A_norm = adj
    else:
        A_norm = adj
    
    # Build the matrix for spectral relaxation
    # Use M = I - alpha * A_norm where alpha controls the strength
    # The leading eigenvector of M will favor vertices in sparse regions
    alpha = 0.5
    M = np.eye(n) - alpha * A_norm
    
    # Power iteration to find leading eigenvector
    # Initialize with a random vector or degree-based initialization
    x = np.ones(n, dtype=float)
    x += np.random.uniform(-0.1, 0.1, n)
    
    # Run power iteration
    max_iter = 50
    for _ in range(max_iter):
        x_new = M @ x
        # Normalize to prevent overflow/underflow
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            break
        x_new = x_new / norm
        # Check convergence
        if np.linalg.norm(x_new - x) < 1e-8:
            x = x_new
            break
        x = x_new
    
    # Ensure non-negative scores by shifting
    x = x - x.min()
    
    # Combine spectral score with degree-based score
    # Lower degree is better, so use -degree as a component
    # Normalize both to [0, 1] range for combination
    
    def normalize(v):
        v_min, v_max = v.min(), v.max()
        if v_max - v_min < 1e-12:
            return np.zeros_like(v)
        return (v - v_min) / (v_max - v_min)
    
    spectral_score = normalize(x)
    degree_score = normalize(-degrees)
    
    # Weighted combination: give more weight to spectral score for global structure
    combined = 0.6 * spectral_score + 0.4 * degree_score
    
    # Return higher scores for vertices we want to pick first
    return combined
# EVOLVE-BLOCK-END
