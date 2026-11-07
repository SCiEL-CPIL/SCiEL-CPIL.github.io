# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Construct a heuristic (edge-attractiveness) matrix for ACO-CVRP.
    
    Strategy: Combine distance-based attractiveness with angular continuity
    bias and demand-aware packing bias.
    """
    n = len(coordinates)
    dm = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    
    if dm.shape != (n, n):
        raise ValueError("distance_matrix shape mismatch")
    
    # Base heuristic: inverse distance squared for steeper gradient
    base = 1.0 / (dm + 1e-9) ** 2
    
    # Demand bias: prefer high-demand nodes to pack trucks efficiently
    alpha = 0.5
    demand_bias = 1.0 + alpha * (dem[np.newaxis, :] / capacity)
    
    # Angular continuity bias: encourage smooth paths
    vec_ij = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]  # coords[j] - coords[i]
    vec_jd = coords - coords[0]  # coords[j] - coords[0] (depot)
    
    vec_ij_norm = np.linalg.norm(vec_ij, axis=2) + 1e-9
    vec_jd_norm = np.linalg.norm(vec_jd, axis=1) + 1e-9
    
    dot = np.sum(vec_ij * vec_jd[np.newaxis, :, :], axis=2)
    cos_theta = dot / (vec_ij_norm * vec_jd_norm[np.newaxis, :])
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    beta = 0.3
    angular_bias = 1.0 + beta * cos_theta
    
    # Combine all biases
    eta = base * demand_bias * angular_bias
    
    # Ensure positivity and finiteness
    eta = np.where(np.isfinite(eta), eta, 1e-9)
    eta = np.maximum(eta, 1e-9)
    np.fill_diagonal(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
