import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s limit
    
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    n = distance_matrix.shape[0]
    
    # Base heuristic: Inverse distance with beta=3.0
    beta = 3.0
    eps = 1e-9
    heuristic = 1.0 / np.power(distance_matrix + eps, beta)
    
    # Angular Consistency Factor
    depot_x, depot_y = coordinates[0]
    x = coordinates[:, 0] - depot_x
    y = coordinates[:, 1] - depot_y
    
    angles = np.arctan2(y, x)
    
    # Pre-compute angular differences for all pairs
    angle_diff = angles[None, :] - angles[:, None]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    
    # Factor: 
    # If |angle_diff| is small, it's a good move (local cluster)
    # If |angle_diff| is large, it's a jump across the map (bad)
    
    factor = np.ones_like(heuristic)
    
    # Mask for small angle diff
    mask_small = np.abs(angle_diff) < (np.pi / 4)
    factor[mask_small] = 1.2
    
    # Mask for large angle diff
    mask_large = np.abs(angle_diff) > (3 * np.pi / 4)
    factor[mask_large] = 0.8
    
    heuristic *= factor
    
    # Lagrangian Relaxation-based edge weight heuristic
    # Solve a relaxed CVRP (ignoring capacity but penalizing load violations)
    # to identify edges that are locally optimal under relaxed constraints
    
    if time.time() < deadline:
        # Multi-start local search to generate diverse good solutions
        # and build a consensus heuristic from edge frequencies
        
        # Initialize edge frequency matrix
        edge_freq = np.zeros_like(heuristic)
        
        # Number of multi-start attempts
        num_starts = min(5, max(2, int((deadline - time.time()) / 5.0)))
        
        # Generate initial solutions using different strategies
        for start_idx in range(num_starts):
            if time.time() > deadline:
                break
            
            # Create initial tour using different heuristics
            if start_idx == 0:
                # Nearest neighbor
                unvisited = set(range(1, n))
                tour = [0]
                current = 0
                while unvisited:
                    # Find nearest unvisited
                    min_dist = np.inf
                    next_node = None
                    for node in unvisited:
                        if distance_matrix[current, node] < min_dist:
                            min_dist = distance_matrix[current, node]
                            next_node = node
                    tour.append(next_node)
                    unvisited.remove(next_node)
                    current = next_node
                tour.append(0)
            elif start_idx == 1:
                # Farthest insertion
                unvisited = set(range(1, n))
                tour = [0]
                current = 0
                while unvisited:
                    # Find farthest unvisited from current route
                    max_dist = -1
                    next_node = None
                    for node in unvisited:
                        # Distance to closest point in current route
                        min_to_route = min(distance_matrix[current, node] for current in tour)
                        if min_to_route > max_dist:
                            max_dist = min_to_route
                            next_node = node
                    tour.append(next_node)
                    unvisited.remove(next_node)
                    current = next_node
                tour.append(0)
            else:
                # Randomized greedy
                unvisited = set(range(1, n))
                tour = [0]
                current = 0
                while unvisited:
                    # Choose randomly with bias toward closer nodes
                    candidates = list(unvisited)
                    dists = [distance_matrix[current, node] for node in candidates]
                    # Soft-min: prefer closer but allow some randomness
                    weights = [1.0 / (d + eps) ** 2 for d in dists]
                    total_w = sum(weights)
                    weights = [w / total_w for w in weights]
                    r = np.random.random()
                    cumsum = 0
                    next_node = candidates[-1]
                    for i, w in enumerate(weights):
                        cumsum += w
                        if r <= cumsum:
                            next_node = candidates[i]
                            break
                    tour.append(next_node)
                    unvisited.remove(next_node)
                    current = next_node
                tour.append(0)
            
            # Local search: 2-opt improvement
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in range(1, len(tour) - 2):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(tour) - 1):
                        # Check if reversing segment i+1 to j improves
                        d1 = distance_matrix[tour[i-1], tour[i]] + distance_matrix[tour[j], tour[j+1]]
                        d2 = distance_matrix[tour[i-1], tour[j]] + distance_matrix[tour[i], tour[j+1]]
                        if d2 < d1 - 1e-10:
                            tour[i:j+1] = tour[i:j+1][::-1]
                            improved = True
            
            # Count edges in the improved tour
            for i in range(len(tour) - 1):
                u, v = tour[i], tour[i+1]
                edge_freq[u, v] += 1
                edge_freq[v, u] += 1
        
        # Normalize edge frequencies and combine with base heuristic
        # Use a power transformation to sharpen the consensus
        max_freq = np.max(edge_freq)
        if max_freq > 0:
            # Scale edge frequencies to [0, 1]
            edge_freq_normalized = edge_freq / max_freq
            
            # Combine: heuristic = base * (1 + alpha * edge_freq_normalized)
            # This boosts edges that appear frequently in good solutions
            alpha = 2.0
            heuristic *= (1.0 + alpha * edge_freq_normalized)
    
    # Ensure positive
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
