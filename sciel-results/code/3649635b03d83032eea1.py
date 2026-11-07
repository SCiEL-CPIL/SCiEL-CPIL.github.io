"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Heuristic matrix for ant colony optimization of orienteering.
    
    Uses a One-Step Lookahead structure:
    Attractiveness of edge (i, j) is proportional to the prize of j,
    inversely proportional to the distance d[i, j], and multiplied by
    the potential prize value reachable from j within the remaining budget.
    
    This biases ants towards nodes that serve as good intermediate steps
    towards other high-value nodes, rather than just the immediate best node.
    """
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

    # 1. Pre-compute feasibility masks
    # min_path_to_i[i] is the shortest distance from 0 to i (which is d[0, i] in Euclidean)
    min_path_to_i = distance_matrix[0, :]
    
    # Feasible[i, j] is True if we can go 0 -> i -> j -> 0 within max_len
    # This is a necessary condition for edge (i, j) to be valid in a tour starting at 0.
    # Note: In reality, path to i might be longer, so this is optimistic.
    # But for heuristic, it's a good filter.
    feasible_mask = (min_path_to_i[:, None] + distance_matrix + distance_matrix[:, 0][None, :]) <= max_len
    
    # 2. Compute Base Score: Prize / Distance
    # We use alpha=1.0 for distance penalty
    alpha = 1.0
    gamma = 0.5 # Exponent for future potential
    
    safe_dist = distance_matrix + eps
    base_score = prizes[None, :] / (safe_dist ** alpha)
    
    # Apply feasibility to base score
    base_score = np.where(feasible_mask, base_score, 0.0)
    
    # 3. Compute Future Potential for each node j
    # P[j] = sum of (prize[k] / d[j, k]^beta) for all k reachable from j
    # Reachable from j means 0 -> j -> k -> 0 is feasible?
    # Let's use a slightly looser constraint for the "potential" to not be too restrictive:
    # d[j, k] + d[k, 0] <= max_len - d[0, j]
    
    # Pre-compute feasible_from_j mask
    # FeasibleFromJ[j, k] is True if d[0, j] + d[j, k] + d[k, 0] <= max_len
    feasible_from_j = (distance_matrix[0, :][None, :] + distance_matrix + distance_matrix[:, 0][None, :]) <= max_len
    
    # Note: feasible_from_j[j, k] is the same logic as feasible_mask[j, k]
    # So we can reuse feasible_mask for the potential calculation, just transposed/logically aligned.
    # feasible_mask[i, j] corresponds to path 0->i->j->0.
    # For potential of j, we look at k reachable from j. The path is 0->j->k->0.
    # So we use feasible_mask[j, k].
    
    beta = 1.0 # Distance penalty for potential
    
    # Potential[j] = sum_k ( prize[k] / d[j, k]^beta * feasible_mask[j, k] )
    # We compute the potential score matrix first
    potential_score_matrix = prizes[None, :] / (safe_dist ** beta)
    potential_score_matrix = np.where(feasible_mask, potential_score_matrix, 0.0)
    
    # Sum over k (axis=1) to get potential for each j
    # Wait, feasible_mask[i, j] is for edge i->j.
    # If we are at j, we consider next node k. The edge is j->k.
    # So we sum over k for fixed j.
    # potential_score_matrix[j, k] is the score of going j->k.
    # Summing over k (axis=1) gives the total potential reachable from j.
    future_potential = np.sum(potential_score_matrix, axis=1)
    
    # 4. Combine Base Score and Future Potential
    # H[i, j] = base_score[i, j] * (future_potential[j] + eps)^gamma
    # This makes sense: moving to j is good if j has high prize (base) AND j leads to other good nodes (potential).
    
    # Add a small constant to future_potential to avoid zeroing out if no nodes are reachable
    # But if no nodes are reachable, the edge is likely infeasible or a dead end.
    # If feasible_mask[i, j] is true, then at least j is reachable.
    
    heuristic = base_score * (future_potential[None, :] + eps) ** gamma
    
    # 5. Refine with "Regret" or "Rank" boost
    # To make the heuristic sharper, we can boost the top choices from each node i.
    # For each i, find the max heuristic value among feasible j.
    # Boost = 1 + boost_factor * (H[i, j] / MaxH[i])
    
    max_heuristic_per_i = np.max(heuristic, axis=1, keepdims=True)
    # Avoid division by zero
    max_heuristic_per_i = np.where(max_heuristic_per_i > 0, max_heuristic_per_i, 1.0)
    
    boost_factor = 1.0
    boost = 1.0 + boost_factor * (heuristic / max_heuristic_per_i)
    
    # Apply boost only where feasible
    heuristic = np.where(feasible_mask, heuristic * boost, eps)
    
    # Ensure all entries are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, eps)
    
    # Set diagonal to eps (self-loops not useful)
    np.fill_diagonal(heuristic, eps)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - start_time:
        pass  # Return what we have

    return heuristic
# EVOLVE-BLOCK-END
