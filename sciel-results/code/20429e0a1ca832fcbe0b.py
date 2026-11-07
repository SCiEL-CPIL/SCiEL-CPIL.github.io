"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    DEADLINE = start_time + 290.0  # 300s limit, keep 10s margin

    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)

    # Ensure shape correctness
    if distance_matrix.shape != (n, n):
        default = np.ones((n, n), dtype=np.float64) * 1e-9
        default[0, 0] = 0.0
        return default

    eps = 1e-9

    # d[j,0]: distance from each node j back to origin
    d_j0 = distance_matrix[:, 0]  # shape (n,)
    min_dist_from_0 = distance_matrix[0, :]  # shape (n,)

    # For each edge (i, j), estimate minimum tour length:
    # min_dist_from_0[i] + d[i,j] + d[j,0]
    min_tour_len = min_dist_from_0[:, None] + distance_matrix + d_j0[None, :]

    # Feasibility mask: edge (i,j) can be part of a valid tour
    feasible = min_tour_len <= max_len

    # Slack: how much budget remains after taking this edge (minimum case)
    slack = max_len - min_tour_len
    slack = np.maximum(slack, 0.0)

    # Base heuristic: prize / (d[i,j] * d[j,0])
    # This penalizes both approach distance and return distance
    numerator = prizes[None, :] + eps
    denominator = (distance_matrix + eps) * (d_j0[None, :] + eps)
    base_heuristic = numerator / denominator

    # Apply feasibility: non-feasible edges get eps
    heuristic = np.where(feasible, base_heuristic, eps)

    # Boost edges with more slack (they are more flexible)
    # Use a mild boost: multiply by (1 + slack / max_len) for feasible edges
    boost = 1.0 + slack / max_len
    boost = np.where(feasible, boost, 1.0)
    heuristic = heuristic * boost

    # Ensure all entries are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, eps)

    # Set diagonal to eps (self-loops not useful)
    np.fill_diagonal(heuristic, eps)

    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - start_time:
        return heuristic

    return heuristic
# EVOLVE-BLOCK-END
