"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Ensure non-negative and finite
    prizes = np.clip(prizes, 0, None)
    distance_matrix = np.clip(distance_matrix, 0, None)
    
    # Strategy: Global Feasibility-Weighted Heuristic
    # The ant's move is valid if len_so_far + d[i,j] + d[j,0] <= max_len.
    # d[j,0] is a fixed cost for visiting node j that must be paid to return.
    # Nodes with high d[j,0] are "expensive" in terms of the path length budget.
    # We want to attract ants to nodes that have high prizes but low "budget cost".
    
    # 1. Calculate the "budget cost" of each node j.
    # The minimum additional length required to visit j and return to 0 is d[j,0].
    # Actually, the constraint is len + d[i,j] + d[j,0].
    # d[j,0] is the part of the cost that is independent of the previous node i (mostly).
    # Let's define cost_j = d[j, 0].
    
    # 2. Normalize the cost relative to max_len.
    # If cost_j is close to max_len, it's very hard to fit other nodes.
    # We use an exponential decay to penalize high costs.
    
    # 3. Combine Prize and Cost.
    # Heuristic h[i,j] should be high if prize[j] is high and cost[j] is low.
    # Note: The heuristic matrix is (N, N). The value h[i,j] depends on j (the destination)
    # and potentially i (the source) if we consider d[i,j].
    # However, the "budget consumption" argument is primarily about j's return cost.
    # Let's refine: The "efficiency" of adding j to the path is prize[j] / (d[i,j] + d[j,0]).
    # But the *global* constraint is that the *sum* of all such segments plus returns must fit.
    # A better proxy for "good node" is prize[j] / d[j,0].
    # Why? Because d[j,0] is the irreducible cost of visiting j in a tour that starts/ends at 0.
    # If d[j,0] is small, you can visit many such nodes.
    
    # Let's try a heuristic that strongly rewards high prize-to-return-distance ratio.
    # h[i,j] = (prize[j] / (d[j,0] + eps))^alpha
    # This makes the heuristic independent of i (source), which simplifies it to a vector broadcast.
    # Is this valid? The ACO rule uses pheromone[i,j] * heuristic[i,j].
    # If heuristic is constant across rows for a given column, it acts as a global bias for node j.
    # This is a valid heuristic structure.
    
    # Let's add a local component too: d[i,j] matters for the immediate step.
    # But the "budget" insight suggests d[j,0] is the dominant constraint driver for *capacity*.
    
    # Let's compute:
    # return_cost = distance_matrix[:, 0]  # shape (N,)
    # prize_efficiency = prizes / (return_cost + 1e-9)
    
    # We want to make high efficiency nodes very attractive.
    # Let's use a power law to amplify the difference.
    # Also, we should ensure that nodes with 0 prize are not attractive.
    
    # Vectorized computation:
    # prize_col = prizes[None, :]  # (1, N)
    # return_cost_col = distance_matrix[:, 0][None, :]  # (1, N)
    # dist_matrix = distance_matrix  # (N, N)
    
    # The current program used: prize[j]^alpha / (d[i,j] + d[j,0])^beta
    # This penalizes d[i,j] heavily.
    # My new strategy: Focus on the *return* cost as the primary constraint.
    # h[i,j] = (prize[j] / (d[j,0] + eps))^alpha * (1 / (d[i,j] + eps))^beta
    # This separates the "global value" (prize/return) from the "local cost" (step distance).
    # This might allow ants to jump to a slightly farther node if it has a much better global efficiency.
    
    alpha = 2.0
    beta = 1.0
    eps = 1e-9
    
    # Calculate global efficiency for each node j
    return_costs = distance_matrix[:, 0]
    # Avoid division by zero
    return_costs_safe = return_costs + eps
    
    # efficiency[j] = prize[j] / return_cost[j]
    # If return_cost is 0 (node 0), prize is 0, so 0/0. We handle node 0 separately or via clip.
    # Prizes[0] is 0.
    
    # We want h[i,j] to be large for good j.
    # Let's construct the matrix.
    
    # Term 1: Global Efficiency
    # eff = prizes / return_costs_safe
    # eff_matrix = eff[None, :] ** alpha  # (1, N) -> (N, N) via broadcast
    
    # Term 2: Local Distance Penalty
    # local_cost = distance_matrix + eps
    # local_penalty = 1.0 / (local_cost ** beta)
    
    # h = eff_matrix * local_penalty
    
    # Let's refine the exponents.
    # If alpha is too high, we might ignore local distance.
    # If beta is too high, we revert to local greedy.
    # The key insight is that d[j,0] is the "tax" on visiting j.
    
    # Alternative:
    # h[i,j] = prize[j]^alpha / (d[i,j] + d[j,0])^beta  (Current)
    # New:
    # h[i,j] = prize[j]^alpha / (d[j,0] + eps)^alpha * 1 / (d[i,j] + eps)^beta
    
    # Let's try this separation.
    
    p = prizes[None, :] ** alpha
    rc = return_costs[None, :] ** alpha
    
    # Handle division by zero for rc (node 0)
    # If rc is 0, and p is 0, result is 0.
    # If rc is 0, and p > 0 (shouldn't happen for node 0 if prize[0]=0), result is inf.
    # We know prize[0]=0.
    
    # Use np.divide to handle zeros safely
    global_eff = np.zeros_like(prizes)
    mask = return_costs > eps
    global_eff[mask] = prizes[mask] / return_costs[mask]
    
    # Now raise to alpha
    global_eff = global_eff ** alpha
    
    # Broadcast to matrix
    heur_global = global_eff[None, :]
    
    # Local penalty
    local_cost = distance_matrix + eps
    heur_local = 1.0 / (local_cost ** beta)
    
    h = heur_global * heur_local
    
    # Ensure positive
    h = np.maximum(h, 1e-9)
    
    return h
# EVOLVE-BLOCK-END
