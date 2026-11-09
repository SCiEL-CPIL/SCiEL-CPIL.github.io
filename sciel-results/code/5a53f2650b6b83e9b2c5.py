# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Heuristic for ant colony orienteering.
    
    Strategy:
    - Combine prize value with a cost-aware denominator.
    - Cost includes distance from current node i to target j, and distance from j back to origin 0.
    - This encourages ants to visit high-prize nodes that are close to the current position
      and also close to the origin (so they can still return within max_len).
    - We use a power law on the cost to amplify the difference between cheap and expensive moves.
    
    The heuristic is static (computed once) and returned as a matrix of the same shape.
    """
    start_time = time.time()
    deadline = start_time + 300.0  # Safety margin before 330s kill
    
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Ensure finite values
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(distance_matrix)):
        return np.ones((n, n), dtype=float) * 1e-9
    
    # cost[i, j] = d[i, j] + d[j, 0]
    # This represents the additional length if we go from i to j and then back to 0
    cost = distance_matrix + distance_matrix[:, 0][None, :]
    
    # prize[j] broadcast over rows
    prize_col = prizes[None, :]  # shape (1, n)
    
    # Use alpha to control how strongly distance penalizes attractiveness
    # Higher alpha -> more preference for close nodes
    alpha = 2.0
    
    # Ensure cost is positive to avoid division by zero
    eps = 1e-9
    cost_safe = np.maximum(cost, eps)
    
    # Heuristic: prize[j] / (cost[i,j])^alpha
    # Nodes with 0 prize (like node 0) will have 0 heuristic, which becomes 1e-9 later
    H = prize_col / (cost_safe ** alpha)
    
    # Check time - if we're close to deadline, return immediately
    if time.time() > deadline - 1.0:
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H
    
    # Refinement: Add a small bonus for nodes that are close to the origin
    # This helps ants prefer nodes that are easy to reach from the start
    # and also easy to return from
    dist_to_origin = distance_matrix[:, 0]  # shape (n,)
    
    # Nodes closer to origin get a slight boost
    # Use a soft bonus: 1 / (1 + dist_to_origin)
    origin_bonus = 1.0 / (1.0 + dist_to_origin[None, :])  # shape (1, n)
    
    # Combine: H = H * (1 + small_factor * origin_bonus)
    # This gently favors nodes near the origin
    small_factor = 0.1
    H = H * (1.0 + small_factor * origin_bonus)
    
    # Ensure no negative or non-finite values
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    # Check time again
    if time.time() > deadline - 1.0:
        return H
    
    # Final safety: ensure shape is correct
    if H.shape != (n, n):
        return np.ones((n, n), dtype=float) * 1e-9
    
    return H
# EVOLVE-BLOCK-END
