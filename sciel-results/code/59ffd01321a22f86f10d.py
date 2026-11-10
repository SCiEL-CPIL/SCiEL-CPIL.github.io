import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for Ant Colony Optimization on the Orienteering Problem.
    
    The heuristic eta[i, j] represents the attractiveness of moving from node i to node j.
    It is designed to favor moves to nodes with high prizes that are both close to the
    current node and close to the origin (to ensure feasibility of returning within max_len).
    
    Formula: eta[i, j] = (prize[j] ** 2) / (distance[i, j] + distance[j, 0] + epsilon)
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    # Distance from each node j to the origin (node 0)
    # Shape: (n,)
    dist_to_origin = distance_matrix[:, 0]
    
    # Distance from node i to node j
    # Shape: (n, n)
    dist_step = distance_matrix
    
    # Construct the denominator: cost to reach j from i + cost to return from j to origin
    # Broadcasting: dist_step (n, n) + dist_to_origin (1, n) -> (n, n)
    denom = dist_step + dist_to_origin[None, :]
    
    # Add a small epsilon to prevent division by zero
    epsilon = 1e-9
    denom += epsilon
    
    # Compute the heuristic
    # Squaring the prize emphasizes high-value nodes, helping the ants focus on the most profitable targets.
    heur = (prizes[None, :] ** 2) / denom
    
    # Ensure the output is a valid numpy array of the correct shape
    # The evaluator handles non-positive values, but we ensure no inf/nan
    heur = np.nan_to_num(heur, nan=1e-9, posinf=1e-9, neginf=1e-9)
    
    return heur
