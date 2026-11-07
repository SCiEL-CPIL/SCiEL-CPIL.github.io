import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    DEADLINE = start_time + 290.0

    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)

    if distance_matrix.shape != (n, n):
        default = np.ones((n, n), dtype=np.float64) * 1e-9
        default[0, 0] = 0.0
        return default

    eps = 1e-9

    d_j0 = distance_matrix[:, 0]  # (n,)
    min_dist_from_0 = distance_matrix[0, :]  # (n,)

    # Minimum tour length through edge (i,j): 0->i->j->0
    min_tour_len = min_dist_from_0[:, None] + distance_matrix + d_j0[None, :]

    feasible = min_tour_len <= max_len
    slack = np.maximum(max_len - min_tour_len, 0.0)

    # Heuristic: prize^1.5 / (d[i,j] + d[j,0])
    # Additive distance matches the constraint structure
    prize_boost = np.power(prizes[None, :] + eps, 1.5)
    dist_sum = distance_matrix + d_j0[None, :] + eps
    base_heuristic = prize_boost / dist_sum

    # Feasibility mask
    heuristic = np.where(feasible, base_heuristic, eps)

    # Mild slack boost
    boost = 1.0 + 0.5 * slack / max_len
    boost = np.where(feasible, boost, 1.0)
    heuristic = heuristic * boost

    # Ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, eps)

    np.fill_diagonal(heuristic, eps)

    return heuristic
