import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute a heuristic matrix for ant colony optimization of CVRP.
    
    Strategy: Use a parameter search over a small grid of heuristic parameters,
    evaluating each candidate with a fast greedy proxy solver. Return the
    heuristic that minimizes the greedy solution cost.
    
    The heuristic form is: eta[i,j] = (1/(d[i,j]+eps))^alpha * demand_weight[j] * depot_bias
    where demand_weight penalizes high-demand nodes and depot_bias encourages
    edges from/to the depot.
    """
    start_time = time.time()
    deadline = start_time + 250  # Safety margin before 330s limit
    
    d = distance_matrix.astype(float)
    n = d.shape[0]
    eps = 1e-9
    
    # Precompute demand as float
    dem = demands.astype(float)
    
    # Greedy proxy solver: given a heuristic matrix, simulate a simple
    # nearest-neighbor with capacity constraint to get a cost estimate
    def greedy_cost(heuristic):
        """Run a greedy construction heuristic and return total distance."""
        total_dist = 0.0
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        current = 0
        load = 0.0
        remaining = n - 1
        
        while remaining > 0:
            # Find best unvisited customer that fits
            best_j = -1
            best_score = -1.0
            for j in range(1, n):
                if visited[j]:
                    continue
                if load + dem[j] > capacity:
                    continue
                # Score: heuristic value
                score = heuristic[current, j]
                if score > best_score:
                    best_score = score
                    best_j = j
            
            if best_j == -1:
                # No customer fits, return to depot
                total_dist += d[current, 0]
                current = 0
                load = 0.0
                continue
            
            total_dist += d[current, best_j]
            load += dem[best_j]
            visited[best_j] = True
            remaining -= 1
            current = best_j
        
        # Return to depot
        total_dist += d[current, 0]
        return total_dist
    
    # Search over parameter combinations
    alpha_values = [0.8, 1.0, 1.2, 1.5]
    beta_values = [0.0, 0.2, 0.4, 0.6, 0.8]
    gamma_values = [0.0, 0.5, 1.0, 2.0]
    
    best_heuristic = None
    best_cost = float('inf')
    
    # Precompute base inverse distance for each alpha
    base_matrices = {}
    for alpha in alpha_values:
        base_matrices[alpha] = (1.0 / (d + eps)) ** alpha
    
    # Try combinations
    for alpha in alpha_values:
        for beta in beta_values:
            for gamma in gamma_values:
                if time.time() > deadline:
                    break
                
                # Build heuristic
                base = base_matrices[alpha]
                
                # Demand weight: lower demand -> higher weight
                # demand_weight[j] = 1 + beta * (1 - dem[j]/capacity)
                demand_weight = 1.0 + beta * (1.0 - dem / capacity)
                demand_weight[0] = 1.0  # Depot
                dw_col = demand_weight.reshape(1, -1)
                
                # Depot bias: edges from depot get boosted
                # This encourages starting routes with nearby customers
                depot_bias = np.ones((n, n))
                depot_bias[0, :] = 1.0 + gamma
                depot_bias[:, 0] = 1.0 + gamma * 0.5  # Slightly less boost for returning
                
                heuristic = base * dw_col * depot_bias
                np.fill_diagonal(heuristic, 0.0)
                
                # Replace non-positive or non-finite
                mask = ~np.isfinite(heuristic) | (heuristic <= 0)
                heuristic[mask] = 1e-9
                
                # Evaluate with greedy proxy
                cost = greedy_cost(heuristic)
                
                if cost < best_cost:
                    best_cost = cost
                    best_heuristic = heuristic.copy()
            
            if time.time() > deadline:
                break
    
    if best_heuristic is None:
        # Fallback: simple inverse distance
        heuristic = 1.0 / (d + eps)
        np.fill_diagonal(heuristic, 0.0)
        mask = ~np.isfinite(heuristic) | (heuristic <= 0)
        heuristic[mask] = 1e-9
        return heuristic
    
    return best_heuristic
