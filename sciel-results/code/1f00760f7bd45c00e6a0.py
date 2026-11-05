import numpy as np

def priority(adjacency):
    # Convert to numpy array if not already
    A = np.asarray(adjacency, dtype=float)
    
    # Compute degrees
    degrees = A.sum(axis=1)
    
    # Avoid division by zero for isolated nodes (though they have no neighbors, cost is 0)
    # We use d+1 to be safe and smooth
    inv_deg = 1.0 / (degrees + 1.0)
    
    # Compute the sum of inverse degrees of neighbors for each vertex
    # This represents the "cost" of selecting that vertex in terms of lost potential neighbors
    costs = A @ inv_deg
    
    # We want to select vertices with LOW cost.
    # The decoder selects in DESCENDING order of score.
    # So we return negative costs.
    # We also subtract a small multiple of degree to prefer lower degree vertices 
    # among those with similar costs, which is a robust fallback.
    # The coefficient 0.1 is a heuristic weight.
    scores = -costs - 0.1 * degrees
    
    return scores
