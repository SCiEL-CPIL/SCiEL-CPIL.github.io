"""Heuristic matrix for ant colony optimization of orienteering."""
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness heuristic matrix for the ant colony.
    
    Design: Greedy-Rank Heuristic.
    1. Construct a greedy tour using a prize-to-total-cost ratio to determine
       a visitation order (rank) for each node.
    2. Use this rank to modulate the base attractiveness (Prize / Step Distance).
       Nodes visited earlier in the greedy tour are considered higher quality
       and receive higher heuristic weights.
    
    This injects global path information into the static heuristic, guiding
    ants towards nodes that form coherent, high-value paths early on.
    """
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    eps = 1e-6
    
    # 1. Greedy Construction to determine node ranks
    # We want to find an order of visiting nodes that maximizes prize density.
    # Metric for greedy step: Prize[j] / (d[current, j] + d[j, 0])
    # This approximates the prize gained per unit of "extension + return" cost.
    
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    dist_so_far = 0.0
    rank = np.full(n, n) # Default rank for unvisited nodes is n (worst)
    step_count = 0
    
    for _ in range(n):
        best_j = -1
        best_score = -1.0
        
        for j in range(n):
            if visited[j]:
                continue
            
            # Check if moving to j and returning to 0 is within budget
            # d[i, j] + d[j, 0] must fit in remaining budget
            # Total length so far: dist_so_far
            # New length: dist_so_far + d[current, j]
            # Return leg: d[j, 0]
            # Constraint: dist_so_far + d[current, j] + d[j, 0] <= max_len
            
            step_cost = distance_matrix[current, j]
            return_cost = distance_matrix[j, 0]
            
            if dist_so_far + step_cost + return_cost <= max_len:
                # Greedy score: Prize per unit of added path length (step + return)
                # Note: We use step + return in denominator to account for the "commitment"
                # of visiting this node.
                score = prizes[j] / (step_cost + return_cost + eps)
                
                if score > best_score:
                    best_score = score
                    best_j = j
        
        if best_j == -1:
            break
        
        visited[best_j] = True
        rank[best_j] = step_count
        dist_so_far += distance_matrix[current, best_j]
        current = best_j
        step_count += 1
    
    # 2. Compute Heuristic Matrix
    # Base attractiveness: Prize[j] / d[i, j]
    # Modulated by rank: 1 / (1 + rank[j])
    # Nodes with lower rank (visited earlier) get higher weight.
    
    # Vectorized computation
    # d_ij is n x n
    d_ij = distance_matrix + eps
    
    # Prize per step distance
    base_heuristic = prizes[None, :] / d_ij
    
    # Rank weight: 1 / (1 + rank[j])
    # rank is 1D array of size n
    rank_weights = 1.0 / (1.0 + rank)
    
    # Combine
    heuristic = base_heuristic * rank_weights[None, :]
    
    # Ensure finite and positive
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Zero out diagonal
    np.fill_diagonal(heuristic, 0.0)
    
    return heuristic
