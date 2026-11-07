import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes the heuristic matrix for the ACO algorithm.
    Combines inverse distance with a savings-based multiplier to guide
    ants towards edges that complete routes efficiently.
    """
    # Ensure inputs are numpy arrays of type float
    d = np.asarray(distance_matrix, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    n = d.shape[0]
    
    # 1. Base Distance Heuristic: 1 / (d^2 + eps)
    # We use a softened denominator (1e-4) to prevent extreme values for very close points,
    # which can cause premature convergence in ACO.
    eps = 1e-4
    dist_heur = 1.0 / (d**2 + eps)
    
    # 2. Savings Heuristic (Clarke-Wright style)
    # S[i, j] = d[i, 0] + d[0, j] - d[i, j]
    # This measures how much distance is saved by visiting i and j consecutively 
    # instead of separately from the depot.
    d_i0 = d[:, 0].reshape(-1, 1)  # (n, 1)
    d_0j = d[0, :].reshape(1, -1)  # (1, n)
    
    S = d_i0 + d_0j - d
    
    # Clip negative savings to 0 (shouldn't happen with Euclidean, but safe)
    S = np.maximum(S, 0)
    
    # Normalize savings to [0, 1] to act as a multiplier
    S_max = np.max(S)
    if S_max > 0:
        S_norm = S / S_max
    else:
        S_norm = np.zeros_like(S)
    
    # 3. Combine Heuristics
    # eta[i, j] = dist_heur[i, j] * (1 + lambda * S_norm[i, j])
    # The savings term boosts attractiveness for edges that are "good" for route construction.
    # Lambda controls the strength of this boost.
    lambda_s = 1.5
    
    multiplier = 1.0 + lambda_s * S_norm
    heuristic = dist_heur * multiplier
    
    # 4. Handle Diagonal and Invalid Values
    # The diagonal represents self-loops, which are invalid moves.
    # The task states: "Heuristic entries that are not positive become 1e-9"
    # We explicitly set diagonal to 0 first, then apply the mask.
    np.fill_diagonal(heuristic, 0.0)
    
    # Mask for non-finite or non-positive values
    mask = ~np.isfinite(heuristic) | (heuristic <= 0)
    heuristic[mask] = 1e-9
    
    return heuristic
