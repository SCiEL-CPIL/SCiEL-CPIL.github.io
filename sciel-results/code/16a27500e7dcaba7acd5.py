import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    # prizes: shape (n,), distance_matrix: shape (n, n)
    # heuristic[i, j] = attractiveness of going from i to j
    # Cost of going to j and returning to 0: d[i,j] + d[j,0]
    # Reward: prize of j
    n = len(prizes)
    d = distance_matrix
    # d[j, 0] for each j, broadcast over i
    return_cost = d[:, 0]  # shape (n,)
    # d[i, j] is d, shape (n, n)
    # total_cost[i, j] = d[i, j] + d[j, 0]
    total_cost = d + return_cost[None, :]  # shape (n, n)
    # heuristic = prizes[j] / (total_cost + eps)
    # prizes[None, :] is shape (1, n), broadcast over i
    eps = 1e-9
    heuristic = (prizes[None, :] + eps) / (total_cost + eps)
    # Ensure non-finite and non-positive become 1e-9
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    return heuristic
