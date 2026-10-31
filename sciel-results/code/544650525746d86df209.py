"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None
_DEADLINE_BUFFER = 10.0  # seconds safety margin before 150s limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 150.0 - _DEADLINE_BUFFER
    
    # Deadline guard: fall back to greedy if time is up
    if time.time() > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Extract submatrices for efficiency
    # d_from_current: distance from current to each unvisited
    d_from_current = distance_matrix[current_node, unvisited_nodes]
    # d_to_dest: distance from each unvisited to destination
    d_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Get coordinates implicitly via distance matrix? 
    # Wait, the interface provides distance_matrix, not coordinates.
    # We cannot compute dot products directly without coordinates.
    # However, we can use the Law of Cosines if we had the distance between current and destination.
    # Let a = d(current, unvisited_i)
    # Let b = d(unvisited_i, destination)
    # Let c = d(current, destination)
    # cos(theta_i) = (a^2 + b^2 - c^2) / (2 * a * b)
    # Minimizing the turn angle is equivalent to maximizing cos(theta_i) (since theta is in [0, pi]).
    # But we also want short edges. A good proxy for "straight line" is maximizing the projection.
    # Alternatively, we can just use a weighted sum of distances which is the current approach's core.
    
    # Let's rethink. The current approach uses d_current + 0.5 * d_dest.
    # A different strategy: "Savings Algorithm" style lookahead or "2-opt" local repair?
    # No, we are constructive.
    
    # Let's try a different scoring function:
    # Score = d_from_current + d_to_dest + lambda * (d_from_current * d_to_dest) / d_current_dest
    # This term is related to the cosine of the angle.
    # If d_current_dest is large, the angle term is small.
    # Actually, minimizing the path length locally is often approximated by:
    # Score = d_from_current + d_to_dest
    # Which is what the current code does (with a weight).
    
    # Let's try a "Regret" based approach more aggressively.
    # Or, let's try to estimate the total tour length more accurately.
    # Current code: est_total = fwd + min_next + next_to_dest + avg_remaining
    # This is O(N^2) per step if we do it for all candidates.
    # We can vectorize the 2-step lookahead.
    
    # Strategy: Vectorized 2-step lookahead for all unvisited nodes.
    # For each candidate i:
    # Cost(i) = d(c, i) + min_{j != i} ( d(i, j) + d(j, dest) ) + R(remaining)
    # R(remaining) is a constant for all i in this step, so we can ignore it for comparison.
    # We need min_{j != i} ( d(i, j) + d(j, dest) ).
    # Let M = distance_matrix[unvisited_nodes, :].
    # Let D_dest = distance_matrix[unvisited_nodes, destination_node].
    # We want for each row i, the min over j != i of M[i, j] + D_dest[j].
    
    unvisited_idx = unvisited_nodes
    # Submatrix of distances between unvisited nodes
    D_unv = distance_matrix[np.ix_(unvisited_idx, unvisited_idx)]
    
    # D_dest is vector of size N_unv
    # We want for each i: min_{j != i} (D_unv[i, j] + D_dest[j])
    # Add D_dest to each row
    D_unv_plus_dest = D_unv + D_dest[None, :]
    
    # Mask the diagonal (j == i) with inf
    np.fill_diagonal(D_unv_plus_dest, np.inf)
    
    # Find the minimum for each row
    min_next_cost = np.min(D_unv_plus_dest, axis=1)
    
    # Total estimated cost for candidate i:
    # d(c, i) + min_next_cost[i]
    # We can also add a term for the return to destination if it's the last step, but min_next_cost already includes d(j, dest).
    # Wait, if n_unvisited == 2, min_next_cost[i] is d(i, other) + d(other, dest).
    # Then total = d(c, i) + d(i, other) + d(other, dest). Correct.
    
    # What about the remaining nodes?
    # The current program adds avg_remaining.
    # We can approximate the remaining cost as:
    # (n_unvisited - 2) * avg_edge_length
    # But this is the same for all candidates, so it doesn't affect the choice.
    # Unless we want to penalize candidates that leave "hard" nodes.
    
    # Let's stick to the vectorized 2-step lookahead.
    # Score = d_from_current + min_next_cost
    scores = d_from_current + min_next_cost
    
    # Tie-breaking: if scores are very close, prefer shorter first edge?
    # Or just take argmin.
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
