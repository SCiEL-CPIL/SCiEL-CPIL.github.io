# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Heuristic for ant colony orienteering.

    Structural change: exponential-decay (Gaussian-like) kernel with adaptive
    per-instance scale, combined with a prize-weighted reachability gate.

    - cost[i,j] = d[i,j] + d[j,0]  (step + return-to-origin)
    - Kernel: exp(-(cost/sigma)^alpha) with sigma adapted to instance geometry
    - Reachability gate: zero out transitions that cannot fit within max_len
    - Prize weighting: multiply by prize[j] to attract high-value nodes
    - Small uniform floor to keep all entries positive
    """
    start_time = time.time()
    deadline = start_time + 290.0

    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    n = len(prizes)

    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(distance_matrix)):
        return np.ones((n, n), dtype=float) * 1e-9

    # cost[i, j] = d[i, j] + d[j, 0]
    cost = distance_matrix + distance_matrix[:, 0][None, :]

    # Adaptive scale: use median of positive costs to set sigma
    positive_costs = cost[cost > 1e-12]
    if len(positive_costs) > 0:
        sigma = np.median(positive_costs) * 0.7
    else:
        sigma = 0.5
    sigma = max(sigma, 0.05)  # avoid too-small sigma

    alpha = 2.0  # quadratic exponent in the Gaussian-like kernel

    # Exponential decay kernel
    ratio = cost / sigma
    kernel = np.exp(-(ratio ** alpha))

    # Reachability gate: if d[i,j] + d[j,0] > max_len, the move is infeasible
    # Ants can only move to j if current_length + d[i,j] + d[j,0] <= max_len
    # Since current_length >= 0, a necessary condition is d[i,j] + d[j,0] <= max_len
    # We use a soft gate: strongly suppress infeasible moves but keep tiny floor
    infeasible = cost > max_len
    # Soft penalty: multiply by a very small factor for infeasible
    reachability = np.where(infeasible, 0.01, 1.0)

    # Prize weighting: higher prize -> more attractive
    prize_col = np.maximum(prizes[None, :], 1e-9)  # shape (1, n)
    # Use sqrt of prize to avoid over-concentration on single highest prize
    prize_weight = np.sqrt(prize_col)

    # Combine
    H = kernel * reachability * prize_weight

    # Add a small origin-proximity bonus for early moves
    dist_to_origin = distance_matrix[:, 0]
    origin_bonus = np.exp(-dist_to_origin[None, :] / sigma)
    H = H * (1.0 + 0.3 * origin_bonus)

    # Normalize: scale so that max is around 1.0
    h_max = np.max(H)
    if h_max > 1e-12:
        H = H / h_max

    # Ensure all entries are positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)

    # Final shape check
    if H.shape != (n, n):
        return np.ones((n, n), dtype=float) * 1e-9

    return H
# EVOLVE-BLOCK-END
