"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a demand-aware heuristic matrix for the Capacitated Vehicle Routing Problem.
    
    The heuristic is defined as:
    eta[i, j] = 1 / ((distance[i, j] + epsilon) * (demand[j] + 1))
    
    This formulation:
    1. Favors shorter edges (standard ACO heuristic).
    2. Penalizes edges leading to high-demand customers. This encourages ants to 
       prioritize low-demand customers, allowing them to pack more customers into 
       a single vehicle trip before hitting the capacity limit. This typically 
       reduces the total number of trips and the overall route distance.
    
    Parameters:
    distance_matrix (np.ndarray): NxN matrix of Euclidean distances.
    coordinates (np.ndarray): Nx2 array of coordinates (unused in this heuristic).
    demands (np.ndarray): Array of N demands. demands[0] is 0 (depot).
    capacity (float): Vehicle capacity (unused in this heuristic).
    
    Returns:
    np.ndarray: NxN matrix of heuristic values (larger is more attractive).
    """
    # Ensure inputs are numpy arrays
    dist = np.asarray(distance_matrix, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    
    # Add epsilon to distance to avoid division by zero on the diagonal (i==j)
    # and to ensure all values are strictly positive.
    epsilon = 1e-9
    safe_dist = dist + epsilon
    
    # We want to penalize high demand. 
    # Using (demand + 1) ensures no division by zero for the depot (demand=0) 
    # and provides a smooth penalty for customers.
    # Broadcasting: dem is (N,), so dem[np.newaxis, :] is (1, N).
    # safe_dist is (N, N).
    # Result is (N, N).
    demand_factor = dem[np.newaxis, :] + 1.0
    
    denom = safe_dist * demand_factor
    
    # Calculate heuristic
    eta = 1.0 / denom
    
    # Ensure all values are finite and positive as per task contract.
    # The construction above guarantees positive finite values, but we add a 
    # safeguard just in case of unexpected input anomalies.
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
