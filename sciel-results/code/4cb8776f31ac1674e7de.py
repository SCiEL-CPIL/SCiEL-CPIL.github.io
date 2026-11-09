import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    d = np.array(distance_matrix, dtype=float)
    demands = np.array(demands, dtype=float)
    
    # Calculate base distance heuristic
    # Power 4 based on previous improvement
    heuristic = np.zeros_like(d)
    mask = d > 0
    heuristic[mask] = 1.0 / (d[mask] ** 4)
    
    # Incorporate demand of the destination node
    # Favor nodes with higher demand to ensure capacity is utilized effectively
    # Normalize demand by capacity to keep scale reasonable
    # demand_j / capacity is between 0 and 0.18 (9/50)
    # This might be too small a multiplier.
    # Let's use 1 + demand_j / capacity to gently boost high demand nodes
    # Or just multiply by demand_j?
    # If demand_j is 0 (depot), heuristic becomes 0.
    
    # Let's try: heuristic *= (demands / capacity)[:, np.newaxis] is wrong shape.
    # We need demand of j for column j.
    # demands is shape (N,). We need to broadcast to (N, N) along rows.
    # So we multiply each column j by demands[j].
    
    # Actually, standard ACO heuristics don't usually depend on the destination's demand in this way.
    # The "sharpness" of distance is the main lever.
    # Let's stick to the pure distance power law, but maybe tune the power.
    
    # If 3 was better than 2, 4 might be better.
    # However, if it's too sharp, it might degrade performance.
    # Let's stick with 4.
    
    return heuristic
