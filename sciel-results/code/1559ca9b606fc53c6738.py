import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    # prizes: shape (n,), distance_matrix: shape (n, n)
    # heuristic[i, j] = attractiveness of going from i to j
    # New Approach: Net Reward = Prize - lambda * Cost
    # This is a difference-based heuristic, distinct from the ratio-based seed.
    n = len(prizes)
    d = distance_matrix
    # d[j, 0] for each j, broadcast over i
    return_cost = d[:, 0]  # shape (n,)
    # d[i, j] is d, shape (n, n)
    # total_cost[i, j] = d[i, j] + d[j, 0]
    total_cost = d + return_cost[None, :]  # shape (n, n)
    
    # Tune lambda to balance prize (0-1) and cost (0-~2.8)
    # A value around 0.2-0.3 seems reasonable to penalize long trips
    # while still valuing high prizes.
    lambda_val = 0.25
    
    # Calculate net reward
    heuristic = prizes[None, :] - lambda_val * total_cost
    
    # Ensure non-finite and non-positive become 1e-9
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    return heuristic
