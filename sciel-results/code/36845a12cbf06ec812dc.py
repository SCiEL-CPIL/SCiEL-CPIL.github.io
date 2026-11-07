"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step of guided local search for TSP.
    
    We apply a penalty to edges based on their cumulative usage count in 
    previous local optima. This encourages the local search to explore 
    new regions of the search space and escape local optima.
    
    The penalty is linear with respect to the usage count, capped at a 
    maximum value to prevent excessive distortion of the distance matrix.
    
    Parameters:
    -----------
    edge_distance : np.ndarray
        True Euclidean distance matrix (symmetric, shape (n, n)).
    local_opt_tour : list or np.ndarray
        The current local optimum tour as a sequence of node indices.
    edge_n_used : np.ndarray
        Symmetric count matrix of how many times each edge has been used
        in previous local optima (including the current one).
        
    Returns:
    --------
    np.ndarray
        Perturbed distance matrix with same shape as edge_distance.
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s kill
    
    n = edge_distance.shape[0]
    d_out = edge_distance.copy()
    
    if n < 2:
        return d_out
    
    # Ensure edge_n_used is a numpy array for vectorized operations
    if not isinstance(edge_n_used, np.ndarray):
        edge_n_used = np.array(edge_n_used)
        
    # Define penalty parameters
    # We use a linear penalty capped at a maximum value.
    # alpha: scaling factor for the penalty
    # cap: maximum count considered for the penalty
    alpha = 0.2
    cap = 5.0
    
    # Calculate penalty matrix
    # penalty = alpha * min(count, cap)
    # This ensures that edges used 5 or more times get the maximum penalty alpha*cap
    # Edges used fewer times get a proportional penalty.
    
    # We apply this to all off-diagonal elements.
    # edge_n_used is symmetric and has 0 on diagonal (presumably).
    
    penalty = np.minimum(edge_n_used, cap) * alpha
    
    # Add penalty to the distance matrix
    d_out += penalty
    
    # Ensure the matrix is symmetric (defensive, should already be symmetric)
    # d_out = (d_out + d_out.T) / 2.0
    
    # Ensure diagonal is zero
    np.fill_diagonal(d_out, 0.0)
    
    # Ensure all values are finite and non-negative
    d_out = np.where(np.isfinite(d_out), d_out, 0.0)
    d_out = np.maximum(d_out, 0.0)
    
    return d_out
# EVOLVE-BLOCK-END
