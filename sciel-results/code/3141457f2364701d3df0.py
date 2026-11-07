"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute edge-attractiveness matrix for ACO on CVRP.
    
    Uses a combination of:
    1. Inverse distance (power-transformed) to encourage short edges
    2. Clarke-Wright savings to encourage route merging
    3. K-NN based gating to restrict search to geometrically plausible neighbors
    
    The heuristic is:
        eta[i, j] = (1 / d[i, j]^alpha) * (1 + beta * savings_norm[i, j]) * gate[i, j]
    """
    start_time = time.time()
    n = distance_matrix.shape[0]
    d = distance_matrix.astype(np.float64)
    
    # Avoid division by zero
    d_safe = np.where(d > 1e-9, d, 1e9)
    
    # Power-transformed inverse distance
    alpha = 1.5
    eta_base = 1.0 / (d_safe ** alpha)
    
    # Clarke-Wright savings
    depot_dist = d[:, 0].copy()
    savings = depot_dist[:, np.newaxis] + depot_dist[np.newaxis, :] - d
    savings_pos = np.maximum(savings, 0.0)
    
    max_savings = savings_pos.max()
    if max_savings > 1e-9:
        savings_norm = savings_pos / max_savings
    else:
        savings_norm = np.zeros_like(savings_pos)
    
    # Combine base with savings
    beta = 1.0
    eta = eta_base * (1.0 + beta * savings_norm)
    
    # K-NN gating: allow edges to nearest neighbors
    customer_coords = coordinates[1:]
    k = 8  # number of neighbors
    
    if len(customer_coords) >= 3:
        try:
            tree = cKDTree(customer_coords)
            
            # Gate matrix initialized to near-zero
            gate = np.full((n, n), 1e-6)
            
            # For each customer, find k nearest neighbors
            for i in range(1, n):
                # Find k nearest neighbors (excluding self)
                dists, indices = tree.query(customer_coords[i - 1], k=min(k, len(customer_coords) - 1))
                for j_idx in indices:
                    j = j_idx + 1  # Convert to global index
                    if j != i:
                        gate[i, j] = 1.0
                        gate[j, i] = 1.0
            
            # Allow depot to connect to all customers
            gate[0, :] = 1.0
            gate[:, 0] = 1.0
            
            # Apply gate
            eta = eta * gate
            
        except Exception:
            pass
    
    # Ensure positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    np.fill_diagonal(eta, 1e-9)
    
    # Time check
    elapsed = time.time() - start_time
    if elapsed > 10:
        d_safe2 = np.where(d > 1e-9, d, 1e9)
        eta_simple = 1.0 / (d_safe2 ** 1.5)
        eta_simple = np.where(np.isfinite(eta_simple) & (eta_simple > 0), eta_simple, 1e-9)
        np.fill_diagonal(eta_simple, 1e-9)
        return eta_simple
    
    return eta
# EVOLVE-BLOCK-END
