# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """Compute edge-attractiveness matrix for ACO on CVRP.
    
    Uses Delaunay triangulation to define a sparse geometric graph,
    then weights edges by inverse distance, demand compatibility,
    and angular consistency to guide ants toward non-crossing,
    capacity-feasible routes.
    """
    n = distance_matrix.shape[0]
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    
    # Start with a base inverse-distance heuristic
    base = 1.0 / (distance_matrix + 1e-9)
    
    # Build Delaunay triangulation on all points (depot + customers)
    try:
        tri = Delaunay(coords)
    except Exception:
        # Fallback if triangulation fails
        return base
    
    # Extract edges from triangulation (each triangle gives 3 edges)
    edges_set = set()
    for simplex in tri.simplices:
        for i in range(3):
            a = int(simplex[i])
            b = int(simplex[(i + 1) % 3])
            if a != b:
                edges_set.add((min(a, b), max(a, b)))
    
    # Also add nearest-neighbor edges (k=5) to ensure connectivity
    # and to handle boundary points
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    _, nn_indices = tree.query(coords, k=6)  # 6 including self
    for i in range(n):
        for j in nn_indices[i, 1:]:
            j = int(j)
            if j != i:
                edges_set.add((min(i, j), max(i, j)))
    
    # Build the heuristic matrix
    heuristic = np.full((n, n), 1e-9, dtype=np.float64)
    
    # For edges in the sparse graph, compute a rich weight
    for (a, b) in edges_set:
        dist_ab = distance_matrix[a, b]
        
        # Base: inverse distance (shorter edges are more attractive)
        w = 1.0 / (dist_ab + 1e-9)
        
        # Demand compatibility bonus:
        # If a route goes from a to b, the remaining capacity after visiting a
        # should be able to accommodate b. We approximate by checking
        # if demand[b] <= capacity - demand[a] (for a simple 2-stop view)
        # More generally, we favor edges where the demand ratio is moderate
        if a == 0 or b == 0:
            # Depot edges: always feasible, slight bonus for lower-demand customers
            if b != 0:
                w *= (1.0 + 0.1 * (capacity - dem[b]) / capacity)
            if a != 0:
                w *= (1.0 + 0.1 * (capacity - dem[a]) / capacity)
        else:
            # Customer-to-customer: check if visiting both is feasible
            # in a 2-stop segment. If dem[a] + dem[b] <= capacity, bonus.
            if dem[a] + dem[b] <= capacity:
                w *= 1.3
            else:
                w *= 0.7
        
        # Angular consistency: for edges that are part of the triangulation,
        # check if the edge direction aligns with local flow
        # (approximate by checking if the edge is short relative to neighbors)
        # This is a soft penalty for long edges in the triangulation
        max_local_dist = np.max(distance_matrix[a, :]) if n > 1 else 1.0
        if dist_ab > 0.5 * max_local_dist:
            w *= 0.5  # Penalize very long edges
        
        heuristic[a, b] = w
        heuristic[b, a] = w
    
    # Ensure diagonal is zero (no self-loops)
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure all off-diagonal entries are positive (replace non-positive with 1e-9)
    mask = heuristic <= 0
    heuristic[mask] = 1e-9
    
    # Normalize to a reasonable scale
    # The heuristic is multiplied with pheromone (starts at 1, decays by 0.9^iter)
    # So we want values in a range that makes sense. Scale so median is ~1.0
    positive_vals = heuristic[heuristic > 1e-8]
    if len(positive_vals) > 0:
        med = np.median(positive_vals)
        if med > 0:
            heuristic = heuristic / med
    
    return heuristic
# EVOLVE-BLOCK-END
