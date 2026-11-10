import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    try:
        demands = np.asarray(demands, dtype=float)
        distance_matrix = np.asarray(distance_matrix, dtype=float)
        coordinates = np.asarray(coordinates, dtype=float)
        
        N = len(demands)
        
        if distance_matrix.shape != (N, N) or coordinates.shape[0] != N:
            return np.full((N, N), 1e-9)

        eps = 1e-9
        d = distance_matrix + eps
        
        # Base heuristic: inverse distance
        inv_dist = 1.0 / d
        
        # Demand scaling: Prioritize high demand nodes
        # "Demand efficiency": High demand nodes are better to serve
        demand_efficiency = 1.0 + demands
        
        # Depot Pressure / Directional Bias
        # Calculate vectors from depot (0) to all nodes
        depot_coords = coordinates[0]
        vectors = coordinates - depot_coords
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0] = 1.0 # Avoid division by zero for depot itself
        
        # Normalize vectors to get directions
        directions = vectors / norms[:, np.newaxis]
        
        # Calculate cosine similarity matrix
        # sim[i, j] is high if i and j are in similar directions from depot
        sim = directions @ directions.T
        
        # Clip similarity to [0, 1] to only reward "forward" moves, penalize or ignore backward
        # Or just use it as a boost.
        # We want to encourage moves that stay in the same "cluster" (direction)
        # Boost factor: 1 + alpha * sim
        alpha = 0.5
        cluster_boost = 1.0 + alpha * np.clip(sim, 0.0, 1.0)
        
        # Initialize eta
        eta = np.zeros((N, N))
        
        # For customer-to-customer and depot-to-customer (j > 0)
        # eta[i, j] = inv_dist[i, j] * demand_efficiency[j] * cluster_boost[i, j]
        for j in range(1, N):
            eta[:, j] = inv_dist[:, j] * demand_efficiency[j] * cluster_boost[:, j]
            
        # For return to depot (j = 0)
        # Return is attractive if the current node has high demand (truck is likely full)
        # Or if the current node is far from depot?
        # Let's stick to demand-based return pressure.
        # Also, if we are moving "away" from depot, returning is bad?
        # Simple: eta[i, 0] = inv_dist[i, 0] * demand_efficiency[i]
        eta[:, 0] = inv_dist[:, 0] * demand_efficiency
        
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
