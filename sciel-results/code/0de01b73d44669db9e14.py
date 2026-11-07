# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of orienteering.
    
    Approach: Greedy Path Simulation
    - For each node j, simulate a greedy tour starting from j, assuming we arrived
      from the depot (0) with cost d[0,j].
    - The simulation picks the next node k that maximizes prize[k] / d[current, k]
      subject to the constraint that we can still return to 0 within max_len.
    - The "Value" of node j is the total prize collected in this simulated greedy path.
    - The heuristic H[i, j] is Value[j] / d[i, j].
    - This captures the sequential nature of the tour and avoids overestimating
      potential by assuming all feasible neighbors can be visited.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    max_len = float(max_len)
    
    n = len(prizes)
    d = distance_matrix
    return_cost = d[:, 0]
    
    # Precompute potential values for each node j
    values = np.zeros(n, dtype=np.float64)
    
    for j in range(n):
        # Cost to reach j from depot
        cost_to_j = d[0, j]
        remaining_budget = max_len - cost_to_j
        
        if remaining_budget < 0:
            values[j] = 0.0
            continue
            
        # If we are at j, we collect prize[j]
        total_prize = prizes[j]
        current_node = j
        visited = np.zeros(n, dtype=bool)
        visited[j] = True
        
        # Greedy simulation
        while True:
            # Find best next node k
            best_k = -1
            best_ratio = 0.0
            
            # Iterate over unvisited nodes
            for k in range(n):
                if visited[k]:
                    continue
                
                # Check if we can go current -> k -> 0 within remaining budget
                # Note: The budget remaining is for the path from current_node to 0.
                # We add d[current, k] to the path, so the new "distance so far" increases.
                # The constraint is: (path_length_so_far + d[current, k] + d[k, 0]) <= max_len
                # We track remaining_budget as max_len - path_length_so_far.
                # So condition is: d[current, k] + d[k, 0] <= remaining_budget
                
                move_cost = d[current_node, k]
                return_cost_k = d[k, 0]
                
                if move_cost + return_cost_k <= remaining_budget:
                    # Calculate ratio: prize per unit distance
                    if move_cost > 1e-9:
                        ratio = prizes[k] / move_cost
                    else:
                        ratio = prizes[k]
                        
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_k = k
                        
            if best_k == -1:
                break
                
            # Move to best_k
            total_prize += prizes[best_k]
            remaining_budget -= d[current_node, best_k]
            current_node = best_k
            visited[best_k] = True
            
        values[j] = total_prize
        
    # Construct Heuristic Matrix
    H = np.zeros((n, n), dtype=np.float64)
    epsilon = 1e-9
    
    # Feasibility mask: edge i->j is feasible if d[i,j] + d[j,0] <= max_len
    # This is a necessary condition.
    feasible_edges = (d + return_cost[None, :] <= max_len)
    
    safe_d = np.where(d > epsilon, d, 1.0)
    
    # H[i,j] = values[j] / d[i,j]
    H_raw = values[None, :] / safe_d
    
    # Apply feasibility mask
    H = np.where(feasible_edges, H_raw, epsilon)
    
    # Ensure positive and finite
    H = np.where(np.isfinite(H), H, epsilon)
    H = np.maximum(H, epsilon)
    
    # Normalize
    max_H = np.max(H)
    min_H = np.min(H)
    
    if max_H > min_H:
        H = epsilon + (H - min_H) / (max_H - min_H) * (1.0 - epsilon)
    else:
        H = np.ones_like(H) * epsilon
        
    return H
# EVOLVE-BLOCK-END
