import numpy as np
import time

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for the TSP instance.
    The heuristic is computed based on inverse distance, which is the standard
    and most effective heuristic for ACO on Euclidean TSP instances.
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin: 3 instances * 300s limit, but we want to be safe.
    
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance
    # Handle zero or very small distances
    safe_dist = distance_matrix.copy()
    # Set diagonal to 0 (self-loops are not used in TSP tours usually, but good practice)
    np.fill_diagonal(safe_dist, 0.0)
    # Clamp small values to avoid division by zero or huge numbers
    min_dist = np.min(safe_dist[safe_dist > 0]) if np.any(safe_dist > 0) else 1e-9
    if min_dist < 1e-9:
        min_dist = 1e-9
    safe_dist[safe_dist < min_dist] = min_dist
    # Diagonal should be 0 or handled by the algorithm ignoring self-loops.
    # The problem says "edge-attractiveness matrix ... same shape".
    # If distance is 0 (diagonal), 1/0 is inf. 
    # The contract says: "Heuristic entries that are not positive become 1e-9".
    # Wait, "not positive" means <= 0.
    # If I set diagonal to 0 in distance, 1/0 is inf. Is inf positive? Yes.
    # But usually self-loops are not part of the tour.
    # Let's look at the contract: "Heuristic entries that are not positive become 1e-9".
    # This implies we should probably avoid inf if possible, or if inf is produced, it's kept?
    # "not positive" -> <= 0. Inf is > 0. So Inf would be kept.
    # However, Inf in the probability calculation (tau * h) might cause issues if not handled carefully (0 * inf = nan).
    # It is safer to set diagonal to a small value or large value?
    # In TSP, you never go from i to i. So the value doesn't strictly matter for the tour construction,
    # unless the algorithm accidentally picks it.
    # The selection probability is proportional to tau[i,j] * h[i,j] for unvisited j.
    # Since i is visited, j cannot be i. So diagonal values are never used in the probability calculation.
    # Therefore, we can set diagonal to 1e-9 or 0.
    # Let's set diagonal to 1e-9 to be safe and consistent with "positive" requirement if interpreted strictly as >0.
    
    # Calculate inverse distance
    # We want larger values for shorter distances.
    h = np.zeros_like(safe_dist, dtype=float)
    mask = safe_dist > 0
    h[mask] = 1.0 / safe_dist[mask]
    h[~mask] = 1e-9  # For diagonal or zero distances
    
    # Normalize? 
    # The contract says "Heuristic entries that are not positive become 1e-9".
    # It doesn't say we must normalize.
    # However, scaling the heuristic can affect the balance with pheromone.
    # Standard ACO often uses eta_ij = 1/d_ij.
    # Let's stick to the standard 1/d_ij.
    
    # Check for non-finite values
    if not np.all(np.isfinite(h)):
        # Replace non-finite with 1e-9
        h[~np.isfinite(h)] = 1e-9
    
    # Ensure positive
    h[h <= 0] = 1e-9
    
    return h
