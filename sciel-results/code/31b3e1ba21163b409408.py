import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = len(prizes)
    
    # Deadline guard
    start_time = time.time()
    
    # Precompute: for each node j, is it feasible to visit from start (node 0) and return?
    # Actually, feasibility depends on the current node i. But we can precompute:
    # For each node j, the minimum "return cost" is d[j,0].
    # An edge (i,j) is feasible if d[i,j] + d[j,0] <= max_len.
    
    # Compute reachability bonus for each node j:
    # How many other nodes k (k != j, k != 0) are reachable from j?
    # i.e., d[j,k] + d[k,0] <= max_len
    reachability_bonus = np.zeros(n)
    for j in range(1, n):
        count = 0
        for k in range(1, n):
            if k == j:
                continue
            if distance_matrix[j, k] + distance_matrix[k, 0] <= max_len:
                count += 1
        reachability_bonus[j] = count
    
    # Normalize reachability bonus to [0, 1]
    max_bonus = np.max(reachability_bonus)
    if max_bonus > 0:
        reachability_bonus = reachability_bonus / max_bonus
    
    # Compute heuristic matrix
    eps = 1e-9
    heuristic = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                heuristic[i, j] = 1e-9
                continue
            
            d_ij = distance_matrix[i, j]
            d_j0 = distance_matrix[j, 0]
            
            # Feasibility check: can we go i->j and then j->0 within max_len?
            # Note: the ant's path length so far is not known at heuristic computation time,
            # but we can at least check if the edge itself is "plausibly" feasible.
            # If d_ij + d_j0 > max_len, then even starting fresh from i, we can't visit j and return.
            # This is a necessary condition (not sufficient, since path length so far adds up).
            
            if d_ij + d_j0 > max_len:
                # Infeasible edge: very low heuristic
                heuristic[i, j] = 1e-9
            else:
                # Feasible edge: use prize * (1 + reachability_bonus) / distance
                # This rewards high-prize nodes that also keep many options open
                prize_j = prizes[j]
                if prize_j <= 0:
                    heuristic[i, j] = 1e-9
                else:
                    # Weight by reachability: nodes that allow visiting more other nodes are better
                    weight = 1.0 + 0.5 * reachability_bonus[j]
                    heuristic[i, j] = (prize_j * weight) / (d_ij + eps)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Check deadline
    if time.time() - start_time > 290:
        # Safety margin: return immediately
        pass
    
    return heuristic
