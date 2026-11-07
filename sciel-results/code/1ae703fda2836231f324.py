"""Heuristic matrix for ant colony optimization of capacitated vehicle routing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute a heuristic matrix using a structural approach based on
    angular continuity (non-crossing routes) and distance-based savings.
    
    Key design:
    - Base: 1/d (standard inverse distance)
    - Angular continuity: reward edges that maintain smooth route direction
    - Savings: reward edges that save depot trips
    - Depot edges: encourage good route starts/ends based on angular position
    """
    start_time = time.time()
    deadline = start_time + 280.0  # 300s limit, 20s safety margin
    
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    dm = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    
    # Base heuristic: inverse distance
    base_heuristic = 1.0 / (dm + 1e-9)
    
    # Compute angular continuity score for customer-customer edges
    # For edge (i,j), compute the angle between vectors (depot->i) and (i->j)
    # Small angles indicate smooth route continuation
    depot_coords = coords[0]
    angular_score = np.zeros((n, n))
    
    for i in range(1, n):
        vec_depot_i = coords[i] - depot_coords
        norm_di = np.linalg.norm(vec_depot_i)
        if norm_di < 1e-10:
            continue
        vec_depot_i = vec_depot_i / norm_di
        
        for j in range(1, n):
            if i == j:
                continue
            vec_i_j = coords[j] - coords[i]
            norm_ij = np.linalg.norm(vec_i_j)
            if norm_ij < 1e-10:
                continue
            vec_i_j = vec_i_j / norm_ij
            
            # Cosine of angle between depot->i and i->j
            cos_angle = np.dot(vec_depot_i, vec_i_j)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            # Reward small angles (smooth continuation), penalize large angles
            # Use a smooth function: score = (1 + cos_angle) / 2, mapped to [0, 1]
            # Then boost: 1 + alpha * score
            angular_score[i, j] = (1.0 + cos_angle) / 2.0
    
    # Compute savings matrix: s(i,j) = d(0,i) + d(0,j) - d(i,j)
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                s = dm[0, i] + dm[0, j] - dm[i, j]
                savings[i, j] = s
                savings[j, i] = s
    
    # Normalize savings to [0, 1]
    max_savings = np.max(savings)
    if max_savings > 0:
        savings_norm = np.clip(savings, 0, max_savings) / max_savings
    else:
        savings_norm = np.zeros_like(savings)
    
    # Combine: base heuristic modulated by angular continuity and savings
    # Angular continuity has stronger weight as it directly affects route quality
    angular_boost = 1.0 + 2.0 * angular_score
    savings_boost = 1.0 + 1.0 * savings_norm
    
    heuristic = base_heuristic * angular_boost * savings_boost
    
    # Handle depot edges
    # Depot-to-customer: shorter distance is better, but also consider
    # how well the customer fits into a route (angular position)
    for i in range(1, n):
        # Angular position of customer i relative to depot
        vec_depot_i = coords[i] - depot_coords
        norm_di = np.linalg.norm(vec_depot_i)
        
        # Average angular continuity score for edges from i to other customers
        if n > 2:
            avg_angular_i = np.mean(angular_score[i, 1:])
        else:
            avg_angular_i = 0.5
        
        # Depot edge boost: customers that are well-connected to others
        depot_boost = 1.0 + 0.5 * avg_angular_i
        
        heuristic[0, i] = base_heuristic[0, i] * depot_boost
        heuristic[i, 0] = base_heuristic[i, 0] * depot_boost
    
    # Ensure diagonal is zero (no self-loops)
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
