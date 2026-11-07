"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute a heuristic matrix for Ant Colony Optimization.
    
    The heuristic H[i, j] represents the attractiveness of moving from node i to node j.
    It is designed to maximize the total prize collected within the max_len constraint.
    
    Strategy:
    1. An edge (i, j) is infeasible if the minimum possible tour length including this edge
       exceeds max_len. The minimum tour length is d[0, i] + d[i, j] + d[j, 0].
    2. For feasible edges, we estimate the value of moving to j as:
       Value(i, j) = (Prize[j] + Sum of (Prize[k] / (d[j, k] + eps)) for all k reachable from j
                      given the remaining budget after reaching j) / (d[i, j] + eps)
       
       The "remaining budget" after reaching j from i (via 0) is:
       Rem = max_len - d[0, i] - d[i, j]
       
       A node k is considered "reachable from j" in this heuristic if:
       d[j, k] + d[k, 0] <= Rem
       This approximates that if we are at j, we can still visit k and return to 0.
       
       We weight future prizes by inverse distance to prioritize close, high-value nodes.
    """
    start_time = time.time()
    DEADLINE = start_time + 290.0  # Keep 10s margin for safety

    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)

    # Ensure shape correctness
    if distance_matrix.shape != (n, n):
        default = np.ones((n, n), dtype=np.float64) * 1e-9
        default[0, 0] = 0.0
        return default

    eps = 1e-9
    heuristic = np.full((n, n), 1e-9, dtype=np.float64)

    # Precompute distance from 0 to all nodes
    d_from_0 = distance_matrix[0, :]
    # Precompute distance from all nodes to 0
    d_to_0 = distance_matrix[:, 0]

    # Precompute reach_cost[j, k] = d[j, k] + d[k, 0]
    # This represents the cost to go from j to k and then back to 0.
    reach_cost = distance_matrix + d_to_0[None, :]

    # Precompute prize divided by distance for future potential calculation
    # We need prize[k] / (d[j, k] + eps). This is not separable easily, so we compute inside loop.
    # However, we can precompute the inverse distances for speed if needed, 
    # but N=50 is small.

    for i in range(n):
        # Check time limit
        if time.time() > DEADLINE:
            break

        # Minimum tour length if we take edge (i, j): d[0, i] + d[i, j] + d[j, 0]
        # d[0, i] is scalar, d[i, j] is vector, d[j, 0] is vector
        min_tour_len = d_from_0[i] + distance_matrix[i, :] + d_to_0
        
        # Feasible edges for row i
        feasible_mask = min_tour_len <= max_len
        
        if not np.any(feasible_mask):
            continue

        # Remaining budget after reaching j from i (via 0)
        # Rem[j] = max_len - d[0, i] - d[i, j]
        remaining_budget = max_len - d_from_0[i] - distance_matrix[i, :]
        
        # We only care about feasible j's
        feasible_indices = np.where(feasible_mask)[0]
        
        for j in feasible_indices:
            # Check if we should stop
            if time.time() > DEADLINE:
                break
            
            rem = remaining_budget[j]
            
            # Find all k such that we can go j -> k -> 0 within remaining budget
            # Condition: d[j, k] + d[k, 0] <= rem
            # reach_cost[j, :] is the vector of d[j, k] + d[k, 0] for all k
            reachable_mask = reach_cost[j, :] <= rem
            
            # Sum of (Prize[k] / (d[j, k] + eps)) for reachable k
            # We exclude j itself from the "future" sum? 
            # The formula adds Prize[j] separately. 
            # If k=j, d[j,j]=0, so Prize[j]/eps is huge. 
            # We should exclude k=j from the future sum to avoid double counting or division by zero issues.
            # Actually, the heuristic is for the edge (i,j). The value is Prize[j] + future.
            # Future is nodes visited *after* j. So k != j.
            reachable_mask[j] = False
            
            if np.any(reachable_mask):
                # Calculate weighted prize sum
                # dist_j_k = distance_matrix[j, reachable_mask]
                # prize_k = prizes[reachable_mask]
                # term = prize_k / (dist_j_k + eps)
                # future_val = np.sum(term)
                
                dists = distance_matrix[j, reachable_mask]
                pzs = prizes[reachable_mask]
                future_val = np.sum(pzs / (dists + eps))
            else:
                future_val = 0.0
            
            total_value = prizes[j] + future_val
            cost = distance_matrix[i, j] + eps
            heuristic[i, j] = total_value / cost

    # Ensure all entries are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Set diagonal to 1e-9 (self-loops are not useful)
    np.fill_diagonal(heuristic, 1e-9)

    return heuristic
# EVOLVE-BLOCK-END
