"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _greedy_eval(distance_matrix, coordinates, demands, capacity, heuristic):
    """
    Evaluates a heuristic by running a greedy construction (Nearest Neighbor)
    and returns the total distance of the resulting tour.
    """
    n = len(demands)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    remaining_cap = capacity
    total_dist = 0.0
    
    # Greedy construction
    for _ in range(n - 1):
        # Find best next node
        best_j = -1
        best_score = -np.inf
        
        # Vectorized calculation for speed, though n=50 is small
        # Score = pheromone(1.0) * heuristic
        # We only consider unvisited nodes that fit in capacity
        scores = heuristic[current].copy()
        
        # Mask visited and infeasible nodes
        feasible_mask = (~visited) & (demands <= remaining_cap)
        scores[~feasible_mask] = -np.inf
        scores[0] = -np.inf # Depot is visited
        
        # If no feasible customer, return to depot
        if not np.any(np.isfinite(scores)):
            total_dist += distance_matrix[current, 0]
            current = 0
            remaining_cap = capacity
            continue
            
        best_j = np.argmax(scores)
        
        # Move to best_j
        total_dist += distance_matrix[current, best_j]
        remaining_cap -= demands[best_j]
        visited[best_j] = True
        current = best_j
        
    # Return to depot
    total_dist += distance_matrix[current, 0]
    
    return total_dist

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Returns an edge-attractiveness matrix (larger is more attractive).
    Uses instance-adaptive selection via fast greedy simulation.
    """
    n = len(demands)
    
    # Generate candidate heuristics
    candidates = []
    
    # 1. Inverse Distance (Baseline)
    # Add small epsilon to avoid division by zero (distance is 0 for i==j)
    d_safe = distance_matrix + 1e-9
    candidates.append(1.0 / d_safe)
    
    # 2. Inverse Distance Squared
    candidates.append(1.0 / (d_safe ** 2))
    
    # 3. Inverse Distance Cubed
    candidates.append(1.0 / (d_safe ** 3))
    
    # 4. Angular Heuristic
    # Favor moves that keep the tour "smooth" relative to the depot
    # Vector from depot to i, vector from i to j
    # Dot product / (|v1| |v2|) -> cos(theta)
    # High cos -> small angle -> good
    # We want larger heuristic for smaller angles
    # Heuristic = 1 + cos(theta) potentially? Or just cos?
    # Let's try: 1 / (1 + angular_penalty) or similar.
    # Simpler: Use the dot product normalized.
    # v_i = coords[i] - coords[0]
    # v_j = coords[j] - coords[i]
    # This is O(N^3) to compute fully if not careful, but N=50 is small.
    # However, we can compute it vectorized.
    
    coords = np.asarray(coordinates)
    depot = coords[0]
    
    # Vectorized angular heuristic calculation
    # For each i, we want a vector h_i[j] for all j.
    # Let's compute for all i, j pairs.
    # v_i = coords[i] - depot  (shape N, 2)
    # v_j = coords[j] - coords[i] (shape N, N, 2)
    
    vi = coords - depot # N, 2
    
    # coords[i] is coords[:, np.newaxis, :] -> N, 1, 2
    # coords[j] is coords[np.newaxis, :, :] -> 1, N, 2
    # v_j = coords[j] - coords[i] -> N, N, 2
    vj = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]
    
    # Dot product vi . vj
    # vi is N, 2 -> N, 1, 2
    # vj is N, N, 2
    # dot -> N, N
    dot = np.sum(vi[:, np.newaxis, :] * vj, axis=2)
    
    # Norms
    norm_vi = np.linalg.norm(vi, axis=1)[:, np.newaxis] # N, 1
    norm_vj = np.linalg.norm(vj, axis=2) # N, N
    
    # Avoid division by zero
    norm_vj_safe = norm_vj + 1e-9
    norm_vi_safe = norm_vi + 1e-9
    
    cos_theta = dot / (norm_vi_safe * norm_vj_safe)
    
    # Heuristic: Larger for smaller angles (cos closer to 1)
    # Map cos_theta [-1, 1] to [0, 1] roughly?
    # Let's use (1 + cos_theta) / 2 + small_epsilon
    # Or just 1 / (1 + (1 - cos_theta)) = 1 / (2 - cos_theta)
    ang_heur = 1.0 / (2.0 - cos_theta + 1e-9)
    candidates.append(ang_heur)
    
    # 5. Nearest Neighbor Bias
    # Find the k-nearest neighbors for each node and give them high weight
    # For each i, find indices of 5 nearest nodes
    # Set heuristic to 10 for those, 1 for others?
    # Or 1/d for those, 1/(10*d) for others?
    # Let's try: 1/d for 3 nearest, 1/(100*d) for rest.
    
    nn_heur = np.zeros_like(distance_matrix)
    for i in range(n):
        if i == 0: continue
        # Find 3 nearest
        d_i = distance_matrix[i].copy()
        d_i[i] = np.inf
        nearest_idx = np.argsort(d_i)[:3]
        # Assign high weight
        for j in nearest_idx:
            nn_heur[i, j] = 1.0 / (distance_matrix[i, j] + 1e-9)
        # Assign low weight to others
        # Initialize all to 1/(100*d)
        nn_heur[i, :] = 1.0 / (100.0 * (distance_matrix[i, :] + 1e-9))
        # Overwrite nearest
        for j in nearest_idx:
            nn_heur[i, j] = 1.0 / (distance_matrix[i, j] + 1e-9)
            
    # Handle depot row (0)
    # For depot, just use 1/d
    nn_heur[0, :] = 1.0 / (distance_matrix[0, :] + 1e-9)
    nn_heur[0, 0] = 1e-9
    
    candidates.append(nn_heur)

    # Evaluate each candidate using greedy simulation
    best_heur = None
    best_score = np.inf
    
    for h in candidates:
        # Ensure validity: no NaNs, Inf, or negatives (though 1/d is positive)
        if not np.all(np.isfinite(h)):
            continue
        if np.any(h <= 0):
            # Clip to small positive
            h = np.clip(h, 1e-9, None)
        
        score = _greedy_eval(distance_matrix, coordinates, demands, capacity, h)
        
        if score < best_score:
            best_score = score
            best_heur = h
            
    # Fallback to inverse distance if something went wrong
    if best_heur is None:
        best_heur = 1.0 / (distance_matrix + 1e-9)
        
    # Final safety check
    if not np.all(np.isfinite(best_heur)):
        return 1.0 / (distance_matrix + 1e-9)
        
    if np.any(best_heur <= 0):
        best_heur = np.clip(best_heur, 1e-9, None)
        
    if best_heur.shape != distance_matrix.shape:
        return 1.0 / (distance_matrix + 1e-9)

    return best_heur
# EVOLVE-BLOCK-END
