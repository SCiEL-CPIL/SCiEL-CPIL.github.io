import numpy as np
from scipy.spatial import Delaunay
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit
    
    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    
    # Initialize heuristic matrix with very low attractiveness
    # The evaluator converts non-positive to 1e-9, so we can use 0.0
    # or small positive numbers for non-triangulation edges.
    # Using 1e-9 directly is safer to avoid any ambiguity, but 0.0 is 
    # effectively 1e-9 after the evaluator's processing.
    # However, the contract says "Heuristic entries that are not positive become 1e-9".
    # So 0.0 becomes 1e-9.
    
    h = np.full((n, n), 0.0, dtype=float)
    
    # Set diagonal to 0 (will become 1e-9, but ants don't visit visited nodes anyway)
    np.fill_diagonal(h, 0.0)
    
    try:
        # Reconstruct points from distance matrix
        # Using MDS (Metric Multidimensional Scaling) or simple embedding
        # Since points are uniform random in unit square, we can use 
        # classical MDS or just assume we can recover coordinates.
        # Actually, we can use the distance matrix directly for Delaunay?
        # No, Delaunay needs coordinates.
        
        # We need to estimate coordinates.
        # For 50 points, we can use a simple embedding.
        # However, MDS might be slow or unstable.
        # Alternative: Use the distance matrix to define "neighbors" directly?
        # Delaunay triangulation is defined on points. 
        # We can approximate the Delaunay graph by finding the K-nearest neighbors 
        # that form a planar graph, but that's complex.
        
        # Let's try to recover coordinates using a simple method.
        # Since it's 50 points, we can use a subset for initial embedding.
        # Or, simpler: The problem states "50 uniform random points in the unit square".
        # We can assume the distance matrix comes from such points.
        # We can use a simple iterative projection or just use the fact that 
        # the Delaunay triangulation is a subgraph of the Gabriel graph.
        
        # Let's use a simpler topological proxy: 
        # For each node i, find the 4 nearest neighbors (typical for 2D Delaunay).
        # But ensuring planarity is hard.
        
        # Better approach: Use the distance matrix to simulate the Delaunay property.
        # An edge (i,j) is in the Delaunay triangulation if there is no point k 
        # inside the circumcircle of triangle (i,j,k). This is hard without coords.
        
        # Let's stick to a robust topological heuristic without explicit coordinates:
        # Use the "nearest neighbor graph" augmented to ensure connectivity.
        # Or, simply use a very sparse K-NN graph.
        
        # Given the constraint of not having coordinates, let's use a 
        # distance-based sparsification that mimics Delaunay density.
        
        # For each node i, select the 5 nearest neighbors.
        # This creates a sparse graph.
        # Assign high heuristic to these edges.
        
        k = 5
        h = np.zeros((n, n), dtype=float)
        
        for i in range(n):
            if time.time() > deadline:
                break
            # Get distances from i to all others
            dists = dm[i].copy()
            dists[i] = np.inf
            # Find k nearest neighbors
            nn_indices = np.argpartition(dists, k)[:k]
            for j in nn_indices:
                h[i, j] = 1.0
                h[j, i] = 1.0
        
        # Ensure the graph is connected? 
        # If not, ACO might get stuck. 
        # But ACO allows any unvisited node, just with low prob.
        # If the graph is disconnected, ants might be forced to take long jumps
        # or get stuck if they start in one component and can't reach others 
        # with reasonable probability. 
        # However, since non-edges have 1e-9 probability, it's very low but non-zero.
        # This might be okay.
        
        # To be safer, we can add a few long-range edges to ensure connectivity.
        # Or, just use K=6 or 7.
        
        # Let's use K=6 to be safer on connectivity.
        h = np.zeros((n, n), dtype=float)
        k = 6
        for i in range(n):
            if time.time() > deadline:
                break
            dists = dm[i].copy()
            dists[i] = np.inf
            nn_indices = np.argpartition(dists, k)[:k]
            for j in nn_indices:
                h[i, j] = 1.0
                h[j, i] = 1.0

        # Normalize? 
        # The evaluator uses pheromone * heuristic.
        # If heuristic is 1.0 for edges and 1e-9 for others, 
        # the ants will strongly prefer the K-NN graph.
        
        # Ensure positive values for the evaluator's check
        h = np.maximum(h, 1e-9)
        
        return h
        
    except Exception:
        # Fallback to simple inverse distance if topological fails
        d = dm + 1e-9
        h = 1.0 / d
        h = np.maximum(h, 1e-9)
        return h
