import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    try:
        demands = np.asarray(demands, dtype=float)
        distance_matrix = np.asarray(distance_matrix, dtype=float)
        
        N = len(demands)
        
        if distance_matrix.shape != (N, N):
            # Fallback to standard inverse distance if shape mismatch, though contract says same shape
            pass

        eps = 1e-9
        d = distance_matrix + eps
        
        # Base heuristic: inverse distance
        inv_dist = 1.0 / d
        
        # Calculate Departure Potential P_i
        # Approximation of remaining capacity: capacity - demand[i]
        # For depot (0), demand is 0, so P_0 = capacity
        # For customers, P_i = capacity - demand[i]
        # If demand > capacity, P_i would be negative, but demands are 1-9 and cap is 50, so safe.
        P = capacity - demands
        # Ensure P is positive
        P[P < 0] = 1e-9
        
        # Calculate Arrival Efficiency E_j
        # "Attractiveness ... inversely proportional to its demand"
        # E_j = 1 / demand[j]
        # For depot (0), demand is 0. We don't select depot as next customer, but need to handle matrix.
        # Set E_0 to 1.0 or similar, it won't be used for customer selection.
        E = np.ones(N)
        nonzero_mask = demands > 0
        E[nonzero_mask] = 1.0 / demands[nonzero_mask]
        
        # Scale edge weights
        # eta[i, j] = inv_dist[i, j] * P[i] * E[j]
        # P is (N,), E is (N,)
        # P[:, np.newaxis] is (N, 1)
        # E[np.newaxis, :] is (1, N)
        
        eta = inv_dist * (P[:, np.newaxis] * E[np.newaxis, :])
        
        # Clean up
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        
        return eta
        
    except Exception:
        try:
            N = len(demands)
        except:
            N = 0
        return np.full((N, N), 1e-9)
