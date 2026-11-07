"""Heuristic matrix for ant colony optimization of orienteering."""
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ACO orienteering.
    
    Uses a dominance-frontier / optionality-aware heuristic:
    - Base efficiency: prize_j / (d_ij + d_j0 + eps)
    - Optionality bonus: rewards moves that keep more future nodes feasible
    - Smoothed via log-sum-exp style soft-max for numerical stability
    
    Returns a matrix of the same shape as distance_matrix.
    """
    start_time = time.time()
    n = len(prizes)
    d = distance_matrix
    eps = 1e-9
    
    # Full detour cost of visiting j from i: d[i,j] + d[j,0]
    detour = d + d[None, 0]  # shape (n, n)
    
    # Base efficiency: prize / detour cost
    # For node 0 (start), prize is 0, so efficiency is 0; we'll handle that
    eff = (prizes[None, :] + eps) / (detour + eps)
    
    # Optionality / dominance frontier:
    # For each edge (i, j), count how many other unvisited nodes k would become
    # infeasible if we DON'T visit j next. This is approximated by:
    # For a given current length L (unknown at heuristic time), the constraint is
    # L + d[i,j] + d[j,0] <= max_len.
    # Since L varies per ant state, we use a static approximation:
    # The "slack" remaining after visiting j is max_len - d[i,j] - d[j,0].
    # More slack means more future options. We reward edges that preserve slack.
    
    # Slack after visiting j from i: max_len - d[i,j] - d[j,0]
    slack = max_len - detour  # shape (n, n)
    slack = np.maximum(slack, 0.0)
    
    # Optionality score: for each (i,j), how many nodes k (k != i, k != j, k != 0)
    # could still be visited after going to j? Approximate by:
    # For each candidate k, the additional cost to visit k after j is d[j,k] + d[k,0].
    # Node k is feasible if slack[i,j] >= d[j,k] + d[k,0].
    # Count such k's. This gives a static "future optionality" measure.
    
    # Build a matrix: for each pair (j, k), the cost to visit k after j is d[j,k] + d[k,0]
    cost_after_j = d + d[None, 0]  # shape (n, n), cost_after_j[j,k] = d[j,k] + d[k,0]
    
    # For each (i, j), count k where cost_after_j[j,k] <= slack[i,j]
    # and k != 0, k != j, k != i
    # We can do this efficiently with broadcasting for small n (n=50)
    
    # Optionality[i,j] = number of k in {1..n-1}, k != i, k != j, such that cost_after_j[j,k] <= slack[i,j]
    optionality = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j == 0:
                continue
            s = slack[i, j]
            if s <= 0:
                continue
            # Count k where cost_after_j[j,k] <= s, k != 0, k != i, k != j
            counts = np.sum((cost_after_j[j, :] <= s) & (np.arange(n) != 0) & (np.arange(n) != i) & (np.arange(n) != j))
            optionality[i, j] = counts
    
    # Combine: heuristic = eff * (1 + optionality)
    # Use a power to amplify differences
    gamma = 1.5
    heuristic = np.power(eff, gamma) * (1.0 + optionality)
    
    # Ensure positivity and finite values
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Normalize for numerical stability: scale so max is around 1
    max_val = np.max(heuristic)
    if max_val > 0:
        heuristic = heuristic / max_val
    
    # Ensure all entries are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Safety: check time budget
    if time.time() - start_time > 280:
        # Return a simple fallback if we're running out of time
        fallback = (prizes[None, :] + eps) / (detour + eps)
        fallback = np.where(np.isfinite(fallback) & (fallback > 0), fallback, 1e-9)
        return fallback
    
    return heuristic
