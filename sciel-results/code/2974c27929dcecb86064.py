# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign a priority score to each vertex for greedy independent set selection.
    
    Approach:
    1. Formulate the priority assignment as a Maximum Weight Independent Set problem.
    2. Use a spectral relaxation (Power Iteration on the normalized adjacency matrix) 
       to generate an initial score vector that captures global structural constraints.
    3. Refine the scores using a fixed-point relaxation that models the marginal loss 
       of selecting neighboring vertices (x_i = max(0, 1 - sum(x_j for j in neighbors))).
    4. Combine the relaxed scores with a degree-based heuristic for tie-breaking.
    """
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([], dtype=float)
    
    degrees = adj.sum(axis=1)
    
    # 1. Spectral Initialization
    # Construct Symmetric Normalized Adjacency Matrix A_sym = D^{-1/2} A D^{-1/2}
    sqrt_deg = np.sqrt(degrees)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_sqrt_deg = 1.0 / sqrt_deg
        inv_sqrt_deg[~np.isfinite(inv_sqrt_deg)] = 0.0
        
    D_inv_sqrt = np.diag(inv_sqrt_deg)
    A_sym = D_inv_sqrt @ adj @ D_inv_sqrt
    
    # Power Iteration to find the leading eigenvector of -A_sym
    # This corresponds to the eigenvector of A_sym associated with the smallest eigenvalue,
    # which often highlights structural boundaries or sparse regions suitable for independent sets.
    M = -A_sym
    x = np.ones(n, dtype=float)
    
    max_iter_spec = 50
    for _ in range(max_iter_spec):
        x_new = M @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            break
        x_new = x_new / norm
        if np.linalg.norm(x_new - x) < 1e-8:
            x = x_new
            break
        x = x_new
        
    # Use absolute value to ensure positive scores for initialization
    spectral_scores = np.abs(x)
    
    # 2. Fixed-Point Relaxation
    # Models the marginal loss: x_i is reduced by the sum of scores of its neighbors.
    # x_{k+1} = max(0, 1 - A x_k)
    # Initialize with spectral scores, normalized to [0, 1]
    max_spec = spectral_scores.max()
    if max_spec > 1e-9:
        x = spectral_scores / max_spec
    else:
        x = np.ones(n, dtype=float)
        
    max_iter_relax = 100
    for _ in range(max_iter_relax):
        # Calculate marginal loss from neighbors
        neighbor_sum = adj @ x
        x_new = 1.0 - neighbor_sum
        x_new = np.maximum(0.0, x_new)
        
        # Check convergence
        if np.linalg.norm(x_new - x) < 1e-6:
            x = x_new
            break
        x = x_new
        
    # 3. Combine with Degree Heuristic
    # Vertices with lower degree are "cheaper" to include.
    # Add a small bonus to lower degree vertices to break ties or adjust ordering.
    max_deg = degrees.max()
    if max_deg > 0:
        deg_norm = degrees / max_deg
    else:
        deg_norm = np.zeros(n)
        
    # Higher score is better. 
    # x is the relaxed independent set potential.
    # (1 - deg_norm) is higher for low degree.
    combined_score = x + 0.1 * (1.0 - deg_norm)
    
    return combined_score
# EVOLVE-BLOCK-END
