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
    3. Second-order neighbor degree (friends-of-friends) to capture structural position
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
    
    # Variance of neighbor degrees (low variance = more uniform neighborhood)
    neighbor_deg_sq_sum = A @ (deg ** 2)
    safe_deg_sq = np.where(deg > 0, deg, 1.0)
    mean_neighbor_deg_sq = neighbor_deg_sq_sum / safe_deg_sq
    mean_neighbor_deg_sq = np.where(deg > 0, mean_neighbor_deg_sq, 0.0)
    var_neighbor_deg = mean_neighbor_deg_sq - mean_neighbor_deg ** 2
    var_neighbor_deg = np.maximum(var_neighbor_deg, 0.0)
    std_neighbor_deg = np.sqrt(var_neighbor_deg)
    
    # Second-order: degree of friends-of-friends
    # A^2 @ 1 gives sum of degrees of neighbors' neighbors (including self via A^2)
    # But A^2[v,u] = number of common neighbors of v and u
    # Let's use: for each v, sum over u in N(v) of deg(u) is already neighbor_deg_sum
    # For second order: (A @ deg) @ A gives for each v: sum_u A[v,u] * deg(u) * sum_w A[u,w]
    # Simpler: just use the mean neighbor degree as a proxy for "neighborhood density"
    
    # Score: we want to pick vertices that are:
    # - Low degree (blocks fewer)
    # - Whose neighbors have high mean degree (picking v is "cheaper")
    # - Whose neighbors have low variance (more predictable, less likely to be picked)
    #
    # Use a smooth surrogate: score = -deg + alpha * mean_neighbor_deg - gamma * std_neighbor_deg
    # The -gamma * std term prefers vertices whose neighbors are more uniform
    # (less likely to have a mix of high/low degree that complicates the greedy choice)
    
    alpha = 0.6
    gamma = 0.15
    score = -deg + alpha * mean_neighbor_deg - gamma * std_neighbor_deg
    
    # Time guard: if we're close to deadline, return early
    if time.time() - t0 > 140:
        return score
    
    return score
# EVOLVE-BLOCK-END
