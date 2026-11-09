# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Multi-scale ensemble heuristic for ant colony orienteering.
    
    Structural change: instead of a single fixed-alpha heuristic, we blend
    three heuristics at different distance sensitivities:
      - H_low:  alpha=1.0  (prize-greedy, tolerates longer detours)
      - H_mid:  alpha=2.0  (balanced)
      - H_high: alpha=4.0  (compact-tour, strongly distance-averse)
    
    The blend weights are computed from the prize distribution:
      - If prizes are concentrated (high variance), weight more toward H_low
        to let ants chase high-value nodes even at moderate distance cost.
      - If prizes are diffuse, weight more toward H_high to keep tours compact.
    
    This gives the ACO a richer landscape to exploit pheromone on.
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
    cost = distance_matrix + distance_matrix[:, 0][None, :]
    
    # prize[j] broadcast over rows
    prize_col = prizes[None, :]  # shape (1, n)
    
    # Ensure cost is positive to avoid division by zero
    eps = 1e-9
    cost_safe = np.maximum(cost, eps)
    
    # Three heuristic scales
    H_low  = prize_col / (cost_safe ** 1.0)
    H_mid  = prize_col / (cost_safe ** 2.0)
    H_high = prize_col / (cost_safe ** 4.0)
    
    # Normalize each to [0, 1] range per column (per destination node)
    # This makes the blend weights meaningful
    def normalize_columns(H):
        col_max = np.max(H, axis=0, keepdims=True)
        col_max = np.maximum(col_max, eps)
        return H / col_max
    
    H_low_n  = normalize_columns(H_low)
    H_mid_n  = normalize_columns(H_mid)
    H_high_n = normalize_columns(H_high)
    
    # Compute blend weights based on prize distribution
    # High prize variance -> more weight on H_low (chase prizes)
    # Low prize variance  -> more weight on H_high (keep compact)
    prize_var = np.var(prizes[1:])  # exclude node 0 (prize=0)
    prize_std = np.sqrt(max(prize_var, 1e-12))
    
    # Map prize_std to weights: 
    # prize_std ~ 0.29 (uniform[0,1]) is typical
    # If prize_std is high, favor H_low
    t = min(max(prize_std / 0.35, 0.0), 1.0)  # normalize to [0,1]
    
    w_low  = 0.3 + 0.5 * t      # 0.3 to 0.8
    w_mid  = 0.4 - 0.2 * t      # 0.4 to 0.2
    w_high = 0.3 - 0.3 * t      # 0.3 to 0.0
    
    # Renormalize weights to sum to 1
    w_sum = w_low + w_mid + w_high
    w_low  /= w_sum
    w_mid  /= w_sum
    w_high /= w_sum
    
    # Blend
    H = w_low * H_low_n + w_mid * H_mid_n + w_high * H_high_n
    
    # Add a small origin-proximity bonus to stabilize early moves
    dist_to_origin = distance_matrix[:, 0]  # shape (n,)
    origin_bonus = 1.0 / (1.0 + dist_to_origin[None, :])  # shape (1, n)
    H = H * (1.0 + 0.15 * origin_bonus)
    
    # Check time
    if time.time() > deadline - 1.0:
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H
    
    # Ensure no negative or non-finite values
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    # Final safety: ensure shape is correct
    if H.shape != (n, n):
        return np.ones((n, n), dtype=float) * 1e-9
    
    return H
# EVOLVE-BLOCK-END
