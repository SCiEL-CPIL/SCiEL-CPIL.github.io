import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    deadline = start_time + 300
    
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    prizes = np.clip(prizes, 0, None)
    distance_matrix = np.clip(distance_matrix, 0, None)
    
    prizes = np.where(np.isfinite(prizes), prizes, 0.0)
    distance_matrix = np.where(np.isfinite(distance_matrix), distance_matrix, 1e9)
    
    eps = 1e-9
    
    return_dist = distance_matrix[:, 0]
    
    # Structural Change:
    # Use a multiplicative cost model: Cost = d_ij * d_j0.
    # This penalizes both the step distance and the return distance simultaneously.
    # It favors nodes that are close to the current node AND close to the start.
    # This encourages "compact" tours, which allows fitting more nodes within max_len.
    # Since prizes are uniform, maximizing node count is key.
    
    # Avoid division by zero for node 0 (return_dist[0]=0) or i=j (d_ij=0)
    # Note: d_ij is 0 for i=j. But ants move to unvisited nodes.
    # However, the matrix is computed for all pairs.
    # If i=j, d_ij=0. If j=0, d_j0=0.
    
    denom = distance_matrix * return_dist[None, :]
    
    # If denom is 0, it means either i==j or j==0 (or both).
    # For j==0, prize is 0. 0/0 -> NaN.
    # For i==j, prize is prize[i]. prize[i] / 0 -> Inf.
    # We must handle these.
    
    # We want heuristic[i, j] to be attractive for valid moves.
    # Valid moves are to unvisited nodes.
    # But the heuristic matrix is static.
    # If we leave Inf for i==j, it might cause issues if an ant somehow considers staying?
    # No, ants move to unvisited nodes.
    # But the matrix values must be finite.
    
    # Let's calculate raw heuristic
    with np.errstate(divide='ignore', invalid='ignore'):
        raw_heuristic = prizes[None, :] / denom
    
    # Replace Inf and NaN
    # If denom is 0 and prize > 0, it's Inf. This happens when i=j (and prize[i]>0).
    # We should probably set diagonal to a low value or 0, as self-loops are not allowed.
    # If denom is 0 and prize == 0, it's NaN. This happens for j=0.
    
    # Set non-finite to 1e-9
    heuristic = np.where(np.isfinite(raw_heuristic), raw_heuristic, 1e-9)
    
    # Ensure positivity
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    if time.time() > deadline:
        return np.ones_like(distance_matrix) * 1e-9

    return heuristic
