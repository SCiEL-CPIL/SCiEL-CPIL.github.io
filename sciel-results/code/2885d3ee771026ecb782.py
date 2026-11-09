import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
        d = np.array(distance_matrix, dtype=float)
        demands = np.array(demands, dtype=float)
        
        # Sharpened distance heuristic with demand-awareness
        # Combine inverse distance with a preference for smaller demands
        # (easier to pack) when capacity is tight
        d_safe = d + 1e-9
        
        # Base: inverse distance cubed for strong local clustering
        base = 1.0 / (d_safe ** 3)
        
        # Demand penalty: prefer nodes with smaller demands relative to capacity
        # This helps ants fill vehicles more efficiently
        # For node j, penalize if demand[j] is large relative to remaining capacity
        # Since we don't know remaining capacity at heuristic computation time,
        # use a soft penalty based on demand/capacity ratio
        demand_ratio = demands / capacity  # shape (n,)
        # For edge i->j, apply penalty based on demand of destination j
        n = d.shape[0]
        demand_penalty = 1.0 / (1.0 + demand_ratio[np.newaxis, :])  # shape (n, n)
        
        # Combine: distance heuristic weighted by demand feasibility
        heuristic = base * demand_penalty
        
        # Set diagonal to 0 to discourage self-loops
        np.fill_diagonal(heuristic, 0.0)
        
        # Ensure any non-positive values become 1e-9
        heuristic[heuristic <= 0] = 1e-9
        
        return heuristic
