import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    
    # 1. Distance to depot for shell penalty
    dist_to_depot = distance_matrix[:, 0]
    
    # 2. Shell penalty: penalize edges that jump between different "shells"
    #    (i.e., nodes at very different distances from the depot)
    diff_depot = np.abs(dist_to_depot[:, np.newaxis] - dist_to_depot[np.newaxis, :])
    
    # 3. Demand bias: encourage edges between high-demand nodes
    #    to promote clustering of heavy loads.
    max_demand = np.max(demands)
    if max_demand > 0:
        norm_demands = demands / max_demand
    else:
        norm_demands = np.zeros(n)
        
    # Bias factor: 1.0 for depot edges, up to 2.0 for high-demand customer edges
    demand_bias = 1.0 + 0.5 * (norm_demands[:, np.newaxis] + norm_demands[np.newaxis, :])
    
    # 4. Combine into effective distance
    #    The shell penalty is added to the actual distance.
    #    We use a coefficient of 0.5 to balance it.
    effective_dist = distance_matrix + 0.5 * diff_depot
    
    # 5. Calculate heuristic
    heuristic = demand_bias / (effective_dist + 1e-9)
    
    return heuristic
