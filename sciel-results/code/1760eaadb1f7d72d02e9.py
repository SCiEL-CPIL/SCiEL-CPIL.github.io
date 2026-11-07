"""Heuristic matrix for ACO-TSP: Multi-start NN with 2-opt refinement and edge frequency boosting."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0


def _two_opt(tour, dist_matrix):
    """Improve a tour using 2-opt heuristic. Returns the improved tour."""
    n = len(tour)
    if n < 4:
        return tour
    improved = True
    # Limit iterations to prevent infinite loops or excessive time usage
    max_iterations = 50
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        if time.time() > _DEADLINE - 15.0:
            return tour
        for i in range(1, n - 1):
            if time.time() > _DEADLINE - 15.0:
                return tour
            for j in range(i + 1, n):
                # Skip adjacent nodes
                if j == i + 1:
                    continue
                # Check if reversing segment [i, j] improves tour
                a = tour[i - 1]
                b = tour[i]
                c = tour[j]
                d = tour[(j + 1) % n]
                
                current_cost = dist_matrix[a, b] + dist_matrix[c, d]
                new_cost = dist_matrix[a, c] + dist_matrix[b, d]
                
                if new_cost < current_cost - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour


def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix (larger = more attractive).
    
    Strategy:
    1. Generate multiple diverse Nearest Neighbor (NN) tours starting from different nodes 
       and with some randomness.
    2. Improve each tour with 2-opt local search.
    3. Count the frequency of edges across all improved tours.
    4. Construct heuristic matrix where edge attractiveness is a combination of 
       inverse distance and a strong boost for edges that appear frequently in the 
       optimized tours. This creates a robust global structural prior.
    """
    # Deadline guard: if we're too close to the limit, return a safe heuristic immediately
    if time.time() > _DEADLINE - 5.0:
        d = np.asarray(distance_matrix, dtype=np.float64)
        return 1.0 / (d + 1e-9) ** 2

    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # If n is very small, simple inverse distance is sufficient
    if n < 10:
        heuristic = 1.0 / (d + 1e-9) ** 2
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Generate multiple diverse tours
    num_tours = 10
    tours = []
    
    for t in range(num_tours):
        if time.time() > _DEADLINE - 20.0:
            break
            
        # Different starting nodes for diversity
        start_node = t % n
        
        # Nearest Neighbor Tour with some randomness
        visited = [False] * n
        tour = [start_node]
        visited[start_node] = True
        current = start_node
        
        for _ in range(n - 1):
            if time.time() > _DEADLINE - 20.0:
                break
            # Find nearest unvisited neighbors, pick randomly from top 3 for diversity
            unvisited = [j for j in range(n) if not visited[j]]
            if not unvisited:
                break
                
            # Sort by distance
            unvisited.sort(key=lambda x: d[current, x])
            
            # Pick randomly from top 3 (or fewer if not enough unvisited)
            top_k = min(3, len(unvisited))
            if t == 0:
                # First tour is pure NN
                next_node = unvisited[0]
            else:
                # Subsequent tours have randomness
                next_node = unvisited[np.random.randint(0, top_k)]
                
            tour.append(next_node)
            visited[next_node] = True
            current = next_node
            
        # Ensure tour is complete
        if len(tour) < n:
            # Fallback: add remaining unvisited nodes in order
            for j in range(n):
                if not visited[j]:
                    tour.append(j)
                    visited[j] = True
                    
        # Improve with 2-opt
        tour = _two_opt(tour, d)
        tours.append(tour)
        
    # If no tours were generated, fall back to simple heuristic
    if not tours:
        heuristic = 1.0 / (d + 1e-9) ** 2
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Count edge frequencies across all tours
    edge_counts = np.zeros((n, n), dtype=np.float64)
    for tour in tours:
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            edge_counts[u, v] += 1
            edge_counts[v, u] += 1
    
    # Base heuristic: inverse distance squared
    base_heuristic = 1.0 / (d + 1e-9) ** 2
    max_base = np.max(base_heuristic)
    if max_base == 0:
        max_base = 1.0
    scaled_base = base_heuristic / max_base
    
    # Normalize edge counts to [0, 1]
    max_count = np.max(edge_counts)
    if max_count > 0:
        normalized_counts = edge_counts / max_count
    else:
        normalized_counts = np.zeros_like(edge_counts)
    
    # Combine: strong boost for edges in optimized tours
    # Edges that appear in multiple good tours get high attractiveness
    # Use a smooth combination: base + boost * count
    boost_factor = 15.0
    final_heuristic = scaled_base + boost_factor * normalized_counts
    
    # Ensure diagonal is 0 (evaluator clamps non-positive to 1e-9)
    np.fill_diagonal(final_heuristic, 0.0)
    
    # Ensure all values are finite and positive (defensive)
    final_heuristic = np.where(np.isfinite(final_heuristic) & (final_heuristic > 0),
                               final_heuristic, 1e-9)
    
    return final_heuristic
# EVOLVE-BLOCK-END
