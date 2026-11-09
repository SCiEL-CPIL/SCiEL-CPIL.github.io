# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Heuristic for ant colony orienteering.
    
    Strategy:
    - Prefer nodes with high prize.
    - Strongly penalize moves that increase the total path length significantly.
    - The cost of moving to j from i is d[i, j] + d[j, 0] (step + return).
    - We use a higher exponent (alpha=3.0) to make the heuristic more "local" 
      and contrasty, which helps the best ant find a compact, high-value route.
    - A small bonus is added for nodes close to the origin to stabilize early moves.
    """
    start_time = time.time()
    deadline = start_time + 300.0

    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Safety check for finite values
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(distance_matrix)):
        return np.ones((n, n), dtype=float) * 1e-9
    
    # cost[i, j] = d[i, j] + d[j, 0]
    # This represents the additional length if we go from i to j and then back to 0
    cost = distance_matrix + distance_matrix[:, 0][None, :]
    
    # prize[j] broadcast over rows
    prize_col = prizes[None, :]  # shape (1, n)
    
    # Use alpha=3.0 to strongly prefer close nodes
    # This makes the heuristic more "greedy" regarding distance, which is 
    # often effective for uniform random point distributions.
    alpha = 3.0
    
    # Ensure cost is positive to avoid division by zero
    eps = 1e-9
    cost_safe = np.maximum(cost, eps)
    
    # Base heuristic: prize[j] / (cost[i,j])^alpha
    H = prize_col / (cost_safe ** alpha)
    
    # Refinement: Add a small bonus for nodes that are close to the origin.
    # This helps in the early stages of the tour (when i=0) to pick nodes
    # that are not only close to start but also easy to return from later,
    # although d[j,0] is already in the cost. The bonus slightly reduces
    # the penalty for nodes very close to 0.
    dist_to_origin = distance_matrix[:, 0]  # shape (n,)
    
    # Soft bonus: 1 / (1 + dist_to_origin)
    # This is 1 for d=0 and decays.
    origin_bonus = 1.0 / (1.0 + dist_to_origin[None, :])  # shape (1, n)
    
    # Combine: H = H * (1 + 0.2 * origin_bonus)
    # This gently favors nodes near the origin, amplifying their attractiveness.
    H = H * (1.0 + 0.2 * origin_bonus)
    
    # Check time
    if time.time() > deadline - 1.0:
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H
    
    # Ensure no negative or non-finite values
    # Non-positive heuristics are set to 1e-9 as per task description
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    # Final safety: ensure shape is correct
    if H.shape != (n, n):
        return np.ones((n, n), dtype=float) * 1e-9
    
    return H
# EVOLVE-BLOCK-END
