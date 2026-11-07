import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    heuristic = np.ones((n, n)) * 1e-9
    
    # Pre-calculate distances from 0 to all nodes
    d_from_0 = distance_matrix[0, :]
    
    for i in range(n):
        # Budget remaining when at node i, assuming we came from 0 directly
        # This is the maximum possible remaining length
        remaining_budget = max_len - d_from_0[i]
        
        if remaining_budget <= 0:
            continue
            
        # Find all valid candidate nodes j from i
        # Condition: d[i, j] + d[j, 0] <= remaining_budget
        # And j != i, j != 0
        valid_candidates = []
        for j in range(n):
            if j == i or j == 0:
                continue
            if distance_matrix[i, j] + distance_matrix[j, 0] <= remaining_budget:
                valid_candidates.append(j)
                
        if not valid_candidates:
            continue
            
        # For each valid candidate j, calculate the heuristic
        # Heuristic is inversely proportional to max prize of other valid candidates
        for j in valid_candidates:
            max_alt_prize = 0.0
            for k in valid_candidates:
                if k != j:
                    if prizes[k] > max_alt_prize:
                        max_alt_prize = prizes[k]
            
            # If no other candidates, max_alt_prize is 0, leading to high heuristic
            # This ensures that if only one node is reachable, it is strongly preferred
            heuristic[i, j] = 1.0 / (max_alt_prize + 1e-9)
            
    # Ensure non-finite values become 1e-9
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
