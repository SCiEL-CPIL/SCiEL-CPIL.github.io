"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Score vertices for greedy Maximum Independent Set.
    Higher score = picked earlier.
    
    Strategy: Use a combination of:
    1. Degree - lower degree = blocks fewer = pick earlier  
    2. Neighborhood quality - if neighbors have high degree, picking this vertex
       is 'cheaper' since those neighbors would be less likely to be picked anyway
    3. Effective blocking cost: how many low-degree (highly pickable) neighbors v blocks
    """
    t0 = time.time()
    n = len(adjacency)
    A = np.asarray(adjacency, dtype=np.float64)
    deg = A.sum(axis=1)
    
    # Mean degree of neighbors
    neighbor_deg_sum = A @ deg
    safe_deg = np.where(deg > 0, deg, 1.0)
    mean_neighbor_deg = neighbor_deg_sum / safe_deg
    mean_neighbor_deg = np.where(deg > 0, mean_neighbor_deg, 0.0)
    
    # Variance of neighbor degrees
    neighbor_deg_sq_sum = A @ (deg ** 2)
    mean_neighbor_deg_sq = neighbor_deg_sq_sum / safe_deg
    mean_neighbor_deg_sq = np.where(deg > 0, mean_neighbor_deg_sq, 0.0)
    var_neighbor_deg = mean_neighbor_deg_sq - mean_neighbor_deg ** 2
    var_neighbor_deg = np.maximum(var_neighbor_deg, 0.0)
    std_neighbor_deg = np.sqrt(var_neighbor_deg)
    
    # Effective blocking cost: for each neighbor u of v, how "pickable" is u?
    # A vertex with low degree is more likely to be picked by the greedy decoder,
    # so blocking it is more costly. Use 1/(1+deg(u)) as a pickability proxy.
    pickability = 1.0 / (1.0 + deg)
    blocking_cost = A @ pickability  # sum of pickability of neighbors
    blocking_cost = np.where(deg > 0, blocking_cost, 0.0)
    
    # Mean pickability of neighbors (normalized)
    mean_pickability = blocking_cost / safe_deg
    mean_pickability = np.where(deg > 0, mean_pickability, 0.0)
    
    # Score: we want to pick vertices that are:
    # - Low degree (blocks fewer)
    # - Whose neighbors have high mean degree (picking v is "cheaper")
    # - Whose neighbors have low mean pickability (neighbors are less likely to be good picks)
    #
    # score = -deg + alpha * mean_neighbor_deg - beta * mean_pickability - gamma * std_neighbor_deg
    
    alpha = 0.8
    beta = 1.2
    gamma = 0.1
    score = -deg + alpha * mean_neighbor_deg - beta * mean_pickability - gamma * std_neighbor_deg
    
    # Time guard
    if time.time() - t0 > 140:
        return score
    
    return score
# EVOLVE-BLOCK-END
