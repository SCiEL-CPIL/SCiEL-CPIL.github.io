"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO on TSP.
    Uses a combination of inverse distance and a structural prior
    derived from a quick 2-opt improvement of a nearest-neighbor tour.
    The structural prior emphasizes edges that appear in the good tour.
    """
    n = distance_matrix.shape[0]
    if n == 0:
        return distance_matrix.copy()
    
    # Safety: ensure finite positive output
    # Base heuristic: inverse distance (stronger for close edges)
    # We use a power to sharpen the preference for short edges
    base = 1.0 / (distance_matrix + 1e-9)
    
    # Compute a quick structural prior using 2-opt on a NN tour
    # This is done quickly (O(n^2) or O(n^2 * iterations)) and cached implicitly
    # since the function is called once per instance.
    
    # 1. Nearest Neighbor Tour
    tour = [0]
    unvisited = set(range(1, n))
    current = 0
    while unvisited:
        # Find nearest unvisited
        nearest = min(unvisited, key=lambda j: distance_matrix[current, j])
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    # 2. 2-opt improvement (limited iterations for speed)
    improved = True
    max_iters = 50  # Limit to keep runtime low
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Check if reversing segment i+1..j improves tour
                # Current edges: (tour[i-1], tour[i]) and (tour[j], tour[j+1] or 0 if j==n-1)
                # New edges: (tour[i-1], tour[j]) and (tour[i], tour[j+1] or 0 if j==n-1)
                a = tour[i-1]
                b = tour[i]
                c = tour[j]
                d = tour[(j+1) % n]
                
                old_cost = distance_matrix[a, b] + distance_matrix[c, d]
                new_cost = distance_matrix[a, c] + distance_matrix[b, d]
                
                if new_cost < old_cost - 1e-10:
                    # Reverse segment i..j
                    tour[i:j+1] = tour[i:j+1][::-1]
                    improved = True
    
    # 3. Build structural prior matrix
    # Edges in the good tour get a boost
    structural = np.zeros_like(distance_matrix)
    for i in range(n):
        u = tour[i]
        v = tour[(i+1) % n]
        structural[u, v] = 1.0
        structural[v, u] = 1.0  # Symmetric for heuristic, though ACO uses directed pheromone
    
    # Combine: base heuristic * (1 + alpha * structural)
    # This keeps the inverse distance scaling but boosts tour edges
    alpha = 2.0
    heuristic = base * (1.0 + alpha * structural)
    
    # Ensure positivity and finiteness
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
