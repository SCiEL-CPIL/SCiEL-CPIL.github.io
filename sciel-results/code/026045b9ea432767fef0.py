"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    """
    Compute heuristic matrix for ACO on TSP.
    Combines a strong inverse-distance power with a greedy-tour boost.
    Uses a time budget to ensure safety.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    n = distance_matrix.shape[0]
    
    # 1. Compute base heuristic: inverse distance squared
    # This emphasizes short edges strongly, which is good for TSP.
    # The power of 2 is a good balance; higher powers can make ants too greedy
    # and get stuck in local minima, but power 2 is a common effective choice.
    dm = distance_matrix + 1e-9
    heuristic = 1.0 / (dm ** 2)
    
    # 2. Compute greedy tour from node 0
    # This provides a structural prior: edges that are part of a good
    # greedy solution should be more attractive.
    unvisited = set(range(1, n))
    current = 0
    tour_edges = []
    
    path = [0]
    for _ in range(n-1):
        # Find nearest unvisited neighbor
        # Using numpy for speed
        distances = distance_matrix[current, list(unvisited)]
        idx = np.argmin(distances)
        best_next = list(unvisited)[idx]
        
        tour_edges.append((current, best_next))
        unvisited.remove(best_next)
        path.append(best_next)
        current = best_next
        
    # Close the tour
    tour_edges.append((current, 0))
    
    # 3. Boost edges in the greedy tour
    # Multiplying by a factor > 1 makes these edges more attractive.
    # A factor of 1.5 is a moderate boost, allowing pheromone to still
    # have influence but giving a strong initial nudge towards good edges.
    # We boost both directions to maintain symmetry in the heuristic prior,
    # which is consistent with the symmetric distance matrix.
    for i, j in tour_edges:
        heuristic[i, j] *= 1.5
        heuristic[j, i] *= 1.5
        
    # 4. Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # 5. Time check (though this should be very fast for n=50)
    elapsed = time.time() - start_time
    if elapsed > 290.0:
        # If we somehow exceeded time, return the base heuristic
        # (which is still valid) rather than risking a timeout in the evaluator
        return 1.0 / (dm ** 2)
        
    return heuristic
# EVOLVE-BLOCK-END
