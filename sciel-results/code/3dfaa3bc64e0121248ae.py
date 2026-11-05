import numpy as np
import time

def priority(adjacency):
    A = np.asarray(adjacency)
    n = A.shape[0]
    degree = A.sum(axis=1).astype(float)
    
    # Sum of neighbor degrees
    sum_neigh_deg = A @ degree  # matrix-vector product
    
    # Mean neighbor degree (avoid div by zero)
    mean_neigh_deg = sum_neigh_deg / np.maximum(degree, 1)
    
    # Smooth nonlinearity: use log-sum-exp style soft-max on neighbor degrees
    # This gives more weight to vertices connected to high-degree neighbors
    # while remaining differentiable
    max_neigh_deg = np.max(mean_neigh_deg) if np.any(mean_neigh_deg > 0) else 0
    # Soft-max: exp(x - max) / sum(exp(x - max)) gives a smooth ranking
    shifted = mean_neigh_deg - max_neigh_deg
    weights = np.exp(shifted)
    soft_score = weights / (np.sum(weights) + 1e-10)
    
    # Quadratic penalty for high degree (stronger than linear)
    # This discourages picking vertices that block many others
    quad_penalty = 0.5 * degree ** 2 / (n + 1)
    
    # Combine: prefer low degree, high mean neighbor degree, with smooth weighting
    score = -degree - quad_penalty + 2.0 * soft_score * n
    
    return score
