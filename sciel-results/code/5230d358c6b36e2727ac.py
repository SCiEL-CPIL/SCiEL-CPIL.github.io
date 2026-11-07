# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Generates an edge-attractiveness matrix using a Delaunay Triangulation strategy.
    
    1. Compute Delaunay Triangulation of all points (Depot + Customers).
    2. Identify edges in the triangulation.
    3. Construct heuristic matrix H:
       - For edges in the Delaunay triangulation: H[i, j] = 1.0 / distance[i, j]
       - For other edges: H[i, j] = 1e-9 (effectively forbidden, but kept positive to avoid division by zero or log issues in ACO)
    
    This enforces a topological constraint that prevents the ACO from forming crossing routes,
    aligning the search space with the low-cost structure of Euclidean TSP/CVRP.
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin

    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    
    # Initialize heuristic with very small values (near zero)
    # We use 1e-9 as the "not positive" threshold replacement, but here we want 
    # non-Delaunay edges to be very unattractive.
    heuristic_matrix = np.full((n, n), 1e-9)
    
    # Handle diagonal (self-loops) - should be 0 or irrelevant, but keep finite
    np.fill_diagonal(heuristic_matrix, 0.0)

    try:
        # Compute Delaunay Triangulation
        # coordinates shape is (n, 2)
        # Delaunay returns simplices. For 2D, simplices are triangles (3 points).
        tri = Delaunay(coordinates)
        
        # Extract edges from simplices
        # Each simplex has 3 vertices: (a, b, c). Edges are (a,b), (b,c), (c,a)
        edges = set()
        for simplex in tri.simplices:
            a, b, c = simplex
            edges.add((a, b))
            edges.add((b, c))
            edges.add((c, a))
            
        # Populate heuristic matrix for Delaunay edges
        # We want higher attractiveness for shorter Delaunay edges.
        # H[i, j] = 1 / d(i, j) is a standard ACO heuristic.
        # For non-Delaunay edges, we keep 1e-9.
        
        # Optimization: Vectorize if possible, but n=51 is small, loop is fine.
        for u, v in edges:
            d = distance_matrix[u, v]
            if d > 0:
                heuristic_matrix[u, v] = 1.0 / d
                heuristic_matrix[v, u] = 1.0 / d
            else:
                # If distance is 0 (duplicate points?), assign a high value?
                # Unlikely in random uniform distribution.
                heuristic_matrix[u, v] = 1000.0
                heuristic_matrix[v, u] = 1000.0

    except Exception:
        # Fallback to inverse distance if Delaunay fails (e.g., degenerate points)
        dist_safe = distance_matrix.copy()
        np.fill_diagonal(dist_safe, 1e6)
        heuristic_matrix = 1.0 / dist_safe
        heuristic_matrix[~np.isfinite(heuristic_matrix)] = 1e-9
        heuristic_matrix[heuristic_matrix <= 0] = 1e-9

    # Ensure all values are positive and finite
    # The task states: "Heuristic entries that are not positive become 1e-9"
    # Our non-Delaunay edges are 1e-9, which is positive.
    # Diagonal is 0.0, which is not positive. The task says "not positive become 1e-9".
    # So diagonal should be 1e-9.
    heuristic_matrix[heuristic_matrix <= 0] = 1e-9
    
    if not np.all(np.isfinite(heuristic_matrix)):
        heuristic_matrix = np.ones_like(distance_matrix) * 1e-9

    # Check time
    if time.time() > deadline:
         # If we are running out of time, just return what we have.
         # The computation above is very fast for n=51.
         pass

    return heuristic_matrix
# EVOLVE-BLOCK-END
