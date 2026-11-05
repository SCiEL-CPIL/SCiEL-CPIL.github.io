import numpy as np
import time

def priority(adjacency):
    """
    Score vertices for greedy Maximum Independent Set.
    
    Strategy: Hub-avoidance via inverse-degree influence.
    
    score(v) = sum_{u in N(v)} (1 / (deg(u) + 1)) - beta * deg(v)
    
    - The first term rewards vertices adjacent to low-degree neighbors
      (low "influence" of neighbors means picking v is cheap in terms
      of blocking valuable independent-set candidates).
    - The second term penalizes high-degree vertices (opportunity cost).
    - Using 1/(deg+1) is a smooth surrogate that avoids division by zero
      and down-weights high-degree hubs, encouraging the greedy decoder
      to pick low-degree vertices that don't block many other low-degree
      vertices.
    """
    adj = np.asarray(adjacency, dtype=np.int64)
    n = adj.shape[0]
    
    # Degrees
    deg = adj.sum(axis=1).astype(float)
    
    # Inverse degree weight for each vertex: 1/(deg+1)
    # This is smooth, bounded in (0, 1], and zero-safe.
    inv_deg = 1.0 / (deg + 1.0)
    
    # Sum of neighbor inverse degrees: for each v, sum_{u in N(v)} inv_deg(u)
    sum_neighbor_inv = adj @ inv_deg
    
    # Penalty coefficient: small enough not to dominate, but breaks ties
    # toward lower-degree vertices.
    beta = 0.1
    
    score = sum_neighbor_inv - beta * deg
    
    return score
