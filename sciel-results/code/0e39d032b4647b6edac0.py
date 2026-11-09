import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for ACO using an Inverse Cost-Efficiency approach.
    This models the heuristic as the marginal utility: prize gained per unit of cost.
    Cost is defined as the distance from current node to next node plus the return distance to start.
    """
    # prizes shape: (N,)
    # distance_matrix shape: (N, N)
    
    # Extract distances from every node j to node 0
    d_to_start = distance_matrix[:, 0] # Shape (N,)
    
    # Effective cost for moving i -> j
    # cost[i, j] = d[i, j] + d[j, 0]
    effective_cost = distance_matrix + d_to_start[None, :]
    
    # Heuristic: Prize / (Cost + Constant)
    # We add a constant C to avoid division by zero and to control the dynamic range.
    # If C is too small, close nodes dominate. If C is large, it flattens to just Prize.
    # A value around 0.5 is reasonable for unit square distances.
    C = 0.5
    
    heuristic = prizes[None, :] / (effective_cost + C)
    
    # Ensure non-negative and finite.
    # If heuristic is 0 (due to prize 0), replace with small epsilon.
    # The task says "Heuristic entries that are not positive become 1e-9".
    
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
