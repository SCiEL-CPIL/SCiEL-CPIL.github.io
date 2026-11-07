"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    d = distance_matrix
    d_to_0 = d[:, 0]  # distance from each node to start
    
    # For each node j, compute the "opportunity cost" of visiting it:
    # After visiting j (with some path length), how much budget remains?
    # The minimum path length to visit j is at least d[j,0] (go and return).
    # So max remaining budget after visiting j is max_len - d[j,0].
    # Nodes k that can still be visited after j must satisfy d[j,k] + d[k,0] <= max_len - d[j,0]
    # i.e., d[j,k] + d[k,0] + d[j,0] <= max_len
    
    # For each j, compute the total prize of nodes k (k != j, k != 0) that remain
    # reachable after visiting j. This is the "future value" if we visit j.
    # If this future value is low, visiting j is a poor choice (high regret).
    
    # Precompute for each j: sum of prizes of k where d[j,k] + d[k,0] <= max_len - d[j,0]
    # Actually, the constraint for visiting k after j is:
    # length_so_far_after_j + d[j,k] + d[k,0] <= max_len
    # length_so_far_after_j >= d[j,0] (minimum is going from 0 to j)
    # So a necessary condition: d[j,0] + d[j,k] + d[k,0] <= max_len
    
    future_prize = np.zeros(n)
    for j in range(1, n):  # skip node 0
        remaining = max_len - d_to_0[j]
        if remaining <= 0:
            future_prize[j] = 0.0
            continue
        # Nodes k that can be visited after j
        for k in range(1, n):
            if k == j:
                continue
            if d[j, k] + d_to_0[k] <= remaining:
                future_prize[j] += prizes[k]
    
    # Heuristic: high prize, low cost, and high future value
    # h[i,j] = prize[j] * (1 + future_prize[j]) / (d[i,j] + d_to_0[j])
    # This rewards nodes that are valuable now AND leave room for other valuable nodes
    
    eps = 1e-9
    cost = d + d_to_0[None, :]  # cost[i,j] = d[i,j] + d[j,0]
    
    # Attractiveness: prize * (1 + future_prize) / cost
    # future_prize[j] captures the "regret" - if visiting j blocks many good nodes, future_prize is low
    attractiveness = (prizes[None, :] * (1.0 + future_prize[None, :])) / (cost + eps)
    
    # Ensure positive and finite
    attractiveness = np.where(np.isfinite(attractiveness) & (attractiveness > 0), attractiveness, 1e-9)
    
    return attractiveness
# EVOLVE-BLOCK-END
