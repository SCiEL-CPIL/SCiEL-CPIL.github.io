"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Compute priority scores for vertices to maximize the size of the
    independent set found by a greedy decoder (descending score, skip neighbors).
    
    Strategy:
    Combine 'effective degree' (degree minus neighborhood density) with a bonus
    for high mean neighbor degree, plus an inverse-degree neighbor penalty.
    
    - Effective degree: vertices with many edges among their neighbors are 'cheaper'
      to include because excluding them doesn't waste as many potential independent set members.
    - Mean neighbor degree: vertices whose neighbors are high-degree 'save' those
      high-degree vertices from being wasted if they were picked instead.
    - Inverse neighbor degree: penalizes vertices adjacent to low-degree nodes
      (which are 'easy' to include and would be blocked).
    
    Score = -(effective_degree) + alpha * mean_neighbor_deg - beta * neighbor_inv_sum
    
    Higher score means we pick it earlier.
    """
    start_time = time.time()
    
    # Convert to numpy array
    A = np.asarray(adjacency, dtype=bool)
    n = A.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Ensure it's a valid square matrix
    if A.shape[0] != A.shape[1]:
        return np.zeros(n)
        
    # 1. Compute degrees
    degree = A.sum(axis=1).astype(float)
    
    # 2. Compute mean neighbor degree
    neighbor_deg_sum = A @ degree
    mean_neighbor_deg = neighbor_deg_sum / np.maximum(degree, 1.0)
    
    # 3. Compute effective degree
    # effective_degree(v) = degree(v) - edges_in_N(v)
    # where edges_in_N(v) is the number of edges in the induced subgraph on N(v)
    
    effective_deg = np.zeros(n, dtype=float)
    
    # Pre-compute neighbor lists
    neighbors = [np.where(A[i, :])[0] for i in range(n)]
    
    for v in range(n):
        # Safety check for time
        if time.time() - start_time > 100.0:
            # Fallback: use just degree and mean_neighbor_deg
            break
            
        nbrs = neighbors[v]
        d_v = len(nbrs)
        
        if d_v > 1:
            # Extract induced subgraph adjacency
            sub_adj = A[np.ix_(nbrs, nbrs)]
            # Sum of sub_adj counts each edge twice
            edges_in_nbr = sub_adj.sum() / 2.0
        else:
            edges_in_nbr = 0.0
            
        effective_deg[v] = degree[v] - edges_in_nbr
    
    # 4. Compute inverse-degree neighbor penalty
    # Vertices adjacent to low-degree nodes are penalized because those
    # low-degree nodes are "easy" to include and would be blocked.
    inv_degrees = 1.0 / (degree + 1.0)
    neighbor_inv_sum = A @ inv_degrees
    
    # 5. Combine scores
    # Primary: minimize effective degree (prefer vertices with sparse neighborhoods)
    # Secondary: bonus for high mean neighbor degree (prefer vertices adjacent to high-degree nodes)
    # Tertiary: penalty for having low-degree neighbors (they're easy to include, don't block them)
    
    alpha = 0.2
    beta = 0.1
    score = -effective_deg + alpha * mean_neighbor_deg - beta * neighbor_inv_sum
    
    return score
# EVOLVE-BLOCK-END
