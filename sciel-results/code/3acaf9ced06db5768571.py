import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for ant colony optimization of orienteering.
    
    Strategy:
    The heuristic eta[i, j] is defined as prize_j / (d(i, j) + d(j, 0)).
    This metric estimates the "value density" of visiting node j from node i,
    considering both the travel cost to j and the mandatory return cost to the
    origin (node 0). This is more effective than just d(i, j) because it
    accounts for the total tour extension caused by visiting j.
    """
    prizes = np.asarray(prizes)
    distance_matrix = np.asarray(distance_matrix)
    
    # d(j, 0) for all j
    d_j0 = distance_matrix[:, 0]
    
    # Denominator: d(i, j) + d(j, 0)
    # distance_matrix is (N, N), d_j0 is (N,). Broadcasting adds d_j0 to each column j.
    denominator = distance_matrix + d_j0[None, :]
    
    # Handle zeros to avoid division by zero or inf
    # If denominator is 0, it means i=0 and j=0 (since d(0,0)=0).
    # We set it to a small positive value.
    denominator = np.where(denominator < 1e-9, 1e-9, denominator)
    
    # Calculate heuristic
    # prizes[None, :] broadcasts to (1, N) -> (N, N)
    numerator = prizes[None, :] + 1e-9
    
    eta = numerator / denominator
    
    # Ensure all values are positive as per requirement
    # "Heuristic entries that are not positive become 1e-9"
    eta = np.where(eta > 0, eta, 1e-9)
    
    return eta
