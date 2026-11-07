import numpy as np

def heuristics(distance_matrix):
    d = np.array(distance_matrix, dtype=np.float64)
    
    # Avoid division by zero on the diagonal
    # The diagonal represents distance from a node to itself, which is 0.
    # In ACO, we never pick a visited node, so the diagonal is not used in probability calculation.
    # However, to be safe and prevent inf/nan, we replace 0 with a small epsilon.
    # Actually, the prompt says "Heuristic entries that are not positive become 1e-9".
    # 0 is not positive, so it will become 1e-9 anyway if we return 0.
    # But 1/0 is inf. So we must handle it.
    
    # Find the minimum non-zero distance to use for normalization
    # Flatten, exclude diagonal, find min
    n = d.shape[0]
    mask = ~np.eye(n, dtype=bool)
    non_zero_dists = d[mask]
    # Filter out any zeros that might be off-diagonal (shouldn't happen with uniform random points)
    non_zero_dists = non_zero_dists[non_zero_dists > 1e-10]
    
    if len(non_zero_dists) == 0:
        # Fallback if something goes wrong
        return np.ones_like(d)
        
    d_min = np.min(non_zero_dists)
    
    # Calculate raw heuristic: 1 / d^2
    # To avoid division by zero, replace 0 with d_min (or epsilon)
    # Since d_min is the smallest distance, using it as a floor for 0s (diagonal)
    # makes the diagonal heuristic 1.0 (if using normalized) or 1/d_min^2 (if not).
    # Let's use a safe distance matrix for calculation.
    d_safe = np.where(d < 1e-10, d_min, d)
    
    # Normalized squared inverse distance
    # eta_ij = (d_min / d_ij)^2
    # This ensures max eta is 1.0 (for the shortest edges).
    # This balances the initial pheromone (1.0) with the heuristic.
    eta = (d_min / d_safe) ** 2
    
    # Ensure no NaNs or Infs
    eta = np.nan_to_num(eta, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    # The prompt says non-positive become 1e-9.
    # Our eta is always positive.
    
    return eta
