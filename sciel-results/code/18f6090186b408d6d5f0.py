"""Heuristic matrix for ant colony optimization of capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute a heuristic matrix using a structural Angular Continuity approach.
    
    Instead of relying solely on distance (which encourages local clusters but 
    can lead to fragmented or crossing routes), this heuristic rewards edges 
    that maintain directional consistency away from the depot. This encourages 
    sector-based, non-crossing routes which are structurally superior for CVRP.
    
    Key components:
    1. Base: 1/d (standard inverse distance).
    2. Angular Boost: Rewards edges (i, j) where the direction i->j is 
       consistent with the radial direction depot->i. This promotes smooth 
       'fan-out' routes.
    3. Depot Edges: Standard distance-based, with a slight demand-based 
       consolidation bias.
    """
    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    demands = np.asarray(demands, dtype=np.float64)
    
    # 1. Base Heuristic: Inverse distance
    # Use 1/d to avoid the extreme steepness of 1/d^2 which can trap ants in 
    # very local minima. 1/d provides a smoother gradient.
    base_heuristic = 1.0 / (dm + 1e-9)
    
    # 2. Angular Continuity Component
    # We want to compute for each pair (i, j) where i, j > 0:
    # cos(theta) = dot(P_i - P_0, P_j - P_i) / (|P_i - P_0| * |P_j - P_i|)
    #
    # Vectorization:
    # Let R_i = P_i - P_0 (vector from depot to i)
    # Let E_ij = P_j - P_i (vector from i to j)
    
    # R shape: (n, 2)
    R = coords - coords[0:1, :]
    
    # E shape: (n, n, 2)
    # E[i, j] = coords[j] - coords[i]
    E = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]
    
    # Norms
    # norm_R shape: (n,)
    norm_R = np.linalg.norm(R, axis=1)
    # norm_E shape: (n, n)
    norm_E = np.linalg.norm(E, axis=2)
    
    # Dot products: dot(R_i, E_ij)
    # R[:, np.newaxis, :] shape (n, 1, 2)
    # E shape (n, n, 2)
    # Result shape (n, n)
    dot_products = np.einsum('ik,jik->ij', R, E)
    
    # Cosine similarity
    # Avoid division by zero
    denom = np.outer(norm_R, norm_E) # shape (n, n)
    denom = np.where(denom > 1e-9, denom, 1e-9)
    
    cos_theta = dot_products / denom
    
    # We only care about customer-customer edges (i>0, j>0, i!=j)
    # Create a mask for customer-customer edges
    mask = np.ones((n, n), dtype=bool)
    mask[0, :] = False
    mask[:, 0] = False
    np.fill_diagonal(mask, False)
    
    # Angular Boost:
    # Reward positive cosine (continuation away from depot)
    # Penalize or ignore negative cosine (turning back)
    # We use a boost factor: 1 + alpha * max(0, cos_theta)
    # This ensures the heuristic is always >= base_heuristic for forward moves.
    
    alpha = 2.0  # Strength of angular bias
    
    angular_boost = np.ones((n, n))
    positive_cos = np.clip(cos_theta, 0, 1)
    angular_boost[mask] = 1.0 + alpha * positive_cos[mask]
    
    # Apply boost to base heuristic
    heuristic = base_heuristic * angular_boost
    
    # 3. Depot Edge Refinement
    # For edges involving the depot (0), the angular logic above is not directly 
    # applicable in the same way (or is masked out). 
    # We want to encourage starting routes with customers that are "central" 
    # to their sector or simply use distance.
    # However, we can add a demand-based consolidation bias for depot edges.
    # Higher demand customers should be slightly more attractive to start/end 
    # routes to ensure capacity utilization.
    
    # Demand bias: (demand[i] / capacity)
    # This is small, so it's a subtle nudge.
    demand_bias = demands / capacity
    
    # Boost depot-to-customer edges (0, i)
    # H[0, i] *= (1 + 0.5 * demand_bias[i])
    # Boost customer-to-depot edges (i, 0)
    # H[i, 0] *= (1 + 0.5 * demand_bias[i])
    
    # Actually, simpler is often better. The angular boost already handles 
    # the structure. Let's just ensure depot edges are not artificially 
    # suppressed. The base 1/d handles them well.
    # We will apply a very slight demand boost to encourage consolidating 
    # high-demand items.
    
    depot_mask = np.zeros((n, n), dtype=bool)
    depot_mask[0, 1:] = True
    depot_mask[1:, 0] = True
    
    # Create a demand boost matrix
    # For edge (i, j), boost based on max(demands[i], demands[j])?
    # Or just for depot edges, boost by the demand of the customer.
    
    demand_boost_matrix = np.zeros((n, n))
    for i in range(1, n):
        demand_boost_matrix[0, i] = 1.0 + 0.5 * demand_bias[i]
        demand_boost_matrix[i, 0] = 1.0 + 0.5 * demand_bias[i]
    
    heuristic *= demand_boost_matrix
    
    # Ensure diagonal is zero (no self-loops)
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Final shape check
    if heuristic.shape != distance_matrix.shape:
         raise ValueError("Heuristic shape mismatch")
        
    return heuristic
# EVOLVE-BLOCK-END
