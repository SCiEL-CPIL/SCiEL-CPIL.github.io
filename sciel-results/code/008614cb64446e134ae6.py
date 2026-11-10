"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute edge-attractiveness matrix using a Constraint-Aware Prize-to-Cost ratio
    with a penalty for consuming a large fraction of the total budget.
    
    Heuristic H[i, j] = (prize[j] / (d[i, j] + d[j, 0])) * (remaining_budget_fraction)^gamma
    
    This favors moves to high-prize nodes that are cheap to visit and return from,
    while strongly penalizing moves that use up most of the available path length,
    thereby preserving capacity for additional nodes.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # Initialize heuristic matrix with a very small value
    heuristic = np.full((n, n), 1e-9, dtype=np.float64)
    
    # Precompute return costs: d[j, 0] for all j
    return_costs = distance_matrix[:, 0]
    
    # Parameters
    gamma = 2.0  # Exponent for the remaining budget penalty
    epsilon = 1e-9
    
    # Compute edge-wise heuristic values
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # Cost of moving from i to j and returning to origin
            cost = distance_matrix[i, j] + return_costs[j]
            
            # If the move exceeds the budget, it's infeasible. 
            # The ACO loop will skip it, but we set heuristic to min value just in case.
            if cost >= max_len:
                continue
                
            # Remaining budget fraction after this move
            # We use (max_len - cost) / max_len to represent the "room left"
            # If cost is close to max_len, this is small.
            remaining_fraction = (max_len - cost) / max_len
            
            # Base value: prize-to-cost ratio
            # Higher prize and lower cost is better
            base_value = prizes[j] / (cost + epsilon)
            
            # Penalty term: favors moves that leave more room
            # If remaining_fraction is 0.5, penalty is 0.25 (for gamma=2)
            # If remaining_fraction is 0.9, penalty is 0.81
            budget_penalty = (remaining_fraction + epsilon) ** gamma
            
            heuristic[i, j] = base_value * budget_penalty
    
    # Normalize the heuristic matrix to [1e-9, 1]
    # This ensures that the heuristic values are on a comparable scale to pheromones
    max_val = np.max(heuristic)
    if max_val > 0:
        heuristic = heuristic / max_val
    
    # Ensure all entries are at least 1e-9
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
