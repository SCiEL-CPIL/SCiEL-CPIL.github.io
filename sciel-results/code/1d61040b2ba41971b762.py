"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes an edge-attractiveness matrix for ACO.
    Uses a greedy path to identify a high-quality reference tour and boosts
    heuristic values for edges in that tour, guiding ants towards a known
    feasible and high-prize solution.
    """
    n = len(prizes)
    if n == 0:
        return np.array([])
    
    # Ensure inputs are numpy arrays for efficient operations
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Initialize heuristic with a baseline ratio: prize / distance
    # Adding epsilon to avoid division by zero
    epsilon = 1e-9
    base_heuristic = (prizes[None, :] + epsilon) / (distance_matrix + epsilon)
    
    # Generate a greedy reference path
    # This path is built by always picking the next node that offers the best
    # prize-to-cost ratio, subject to the constraint that the ant can still
    # return to the depot.
    current_node = 0
    current_length = 0.0
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    
    # List of edges (i, j) in the greedy path
    greedy_edges = []
    
    for _ in range(n - 1):
        best_node = -1
        best_score = -np.inf
        
        # Find the best next node
        for j in range(1, n):
            if visited[j]:
                continue
            
            d_ij = distance_matrix[current_node, j]
            d_j0 = distance_matrix[j, 0]
            
            # Check if moving to j allows return to 0
            if current_length + d_ij + d_j0 <= max_len:
                # Score: prize per unit of additional tour length
                # We use d_ij + d_j0 in denominator to account for the 
                # "detour" cost of visiting j and then going back to 0 
                # (though the ant doesn't go back yet, it reserves that budget)
                # Actually, the constraint is on the total path. 
                # A better greedy score is prize / (d_ij + d_j0) 
                # because d_j0 is the "sunk cost" for the return leg 
                # that gets consumed by visiting j.
                score = prizes[j] / (d_ij + d_j0 + epsilon)
                
                if score > best_score:
                    best_score = score
                    best_node = j
        
        if best_node == -1:
            break
        
        greedy_edges.append((current_node, best_node))
        current_length += distance_matrix[current_node, best_node]
        current_node = best_node
        visited[best_node] = True
    
    # Apply boost to heuristic for edges in the greedy path
    # We multiply the base heuristic by a factor for these edges
    # to strongly attract ants to this path initially.
    boost_factor = 10.0
    
    for i, j in greedy_edges:
        base_heuristic[i, j] *= boost_factor
        
        # Also boost the reverse edge? 
        # ACO is directed in terms of path construction, but distance is symmetric.
        # The ant moves i->j. The heuristic eta[i,j] is used.
        # We don't necessarily need to boost j->i unless we expect backtracking,
        # which is not allowed (unvisited nodes only).
        
    # Ensure all values are positive and finite
    base_heuristic = np.where(np.isfinite(base_heuristic), base_heuristic, 1e-9)
    base_heuristic = np.maximum(base_heuristic, 1e-9)
    
    return base_heuristic
# EVOLVE-BLOCK-END
