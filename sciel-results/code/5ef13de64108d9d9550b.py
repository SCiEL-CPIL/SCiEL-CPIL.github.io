import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    dm = np.asarray(distance_matrix, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    
    # Inverse distance with gamma=2, weighted by (1 + demand[j]) to favor
    # high-demand destinations (better packing efficiency)
    # eta[i,j] = d(i,j)^-2 * (1 + demand[j])
    
    # d(i,j)^-2
    d_inv2 = 1.0 / (dm + 1e-9) ** 2
    
    # Demand weight: (1 + demand[j]) for each destination j
    demand_weight = 1.0 + dem  # shape (n,)
    # Broadcast to (n, n): each column j gets weight (1 + demand[j])
    eta = d_inv2 * demand_weight[np.newaxis, :]
    
    # Ensure positive; replace non-positive with 1e-9
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
