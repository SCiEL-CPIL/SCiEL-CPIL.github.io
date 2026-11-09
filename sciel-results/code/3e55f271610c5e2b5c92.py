import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(coordinates)
    eta = np.zeros((n, n))
    
    # Demand-aware heuristic:
    # Favors moving to customers with higher demand.
    # This encourages packing routes with high-demand items, potentially creating denser routes
    # and reducing the number of trips, which helps minimize total distance.
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Base attractiveness is inverse distance
                base = 1.0 / (distance_matrix[i, j] + 1e-9)
                # Multiply by a demand factor to boost attractiveness of high-demand nodes
                # demands are floats (1.0 to 9.0), so factor ranges from 2.0 to 10.0
                demand_factor = 1.0 + demands[j]
                eta[i, j] = base * demand_factor
            else:
                eta[i, j] = 0.0
                
    return eta
