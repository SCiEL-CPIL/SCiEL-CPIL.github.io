# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Return an edge-attractiveness matrix for ant colony orienteering.
    
    Strategy: Combine prize value of destination with distance penalty,
    but also consider the 'detour cost' - how much extra distance is added
    by going through this edge vs going directly. We want edges that lead
    to high-value nodes without adding too much distance.
    
    For each edge (i, j), the heuristic should be high if:
    1. Node j has a high prize
    2. The edge is short (low distance)
    3. The edge doesn't create a large detour (i.e., d[i,j] + d[j,0] is not 
       much larger than d[i,0])
    
    We compute:
    - prize_component: prizes[j] (high prize is good)
    - distance_component: 1/(d[i,j] + eps) (shorter is better)
    - detour_penalty: penalize edges where d[i,j] + d[j,0] >> d[i,0]
    
    Combined heuristic = prize_component * distance_component * detour_factor
    where detour_factor accounts for how 'efficient' the move is.
    """
    n = len(prizes)
    eps = 1e-9
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Prize component: prize of destination node j
    prize_component = prizes[None, :]  # shape (1, n), broadcast to (n, n)
    
    # Distance component: inverse distance from i to j
    dist_inv = 1.0 / (distance_matrix + eps)  # shape (n, n)
    
    # Detour factor: For each edge (i, j), compare the path i->j->0 vs i->0
    # If d[i,j] + d[j,0] is close to d[i,0], the move is efficient.
    # If it's much larger, it's a bad detour.
    # detour_cost[i,j] = d[i,j] + d[j,0] - d[i,0]
    # We want to penalize large detour costs.
    
    # d[i,0] for all i: shape (n,)
    d_i0 = distance_matrix[:, 0]  # shape (n,)
    
    # d[j,0] for all j: shape (n,)
    d_j0 = distance_matrix[:, 0]  # shape (n,)
    
    # detour_cost[i,j] = distance_matrix[i,j] + d_j0[j] - d_i0[i]
    # Shape: (n, n)
    detour_cost = distance_matrix + d_j0[None, :] - d_i0[:, None]
    
    # Clamp detour_cost to be non-negative (triangle inequality should hold, 
    # but numerical issues might cause small negatives)
    detour_cost = np.maximum(detour_cost, 0.0)
    
    # Detour factor: exp(-alpha * detour_cost) where alpha controls sensitivity
    # Larger alpha means more penalty for detours
    alpha = 1.0 / max_len  # Scale by max_len for normalization
    detour_factor = np.exp(-alpha * detour_cost)
    
    # Combine all components
    # heuristic[i,j] = prize_component[j] * dist_inv[i,j] * detour_factor[i,j]
    heuristic = prize_component * dist_inv * detour_factor
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, eps)
    
    # Scale to have reasonable magnitude
    # The pheromone starts at 1, so heuristic should be in a similar range
    # Normalize so that the mean is around 1
    mean_val = np.mean(heuristic)
    if mean_val > eps:
        heuristic = heuristic / mean_val
    
    return heuristic
# EVOLVE-BLOCK-END
