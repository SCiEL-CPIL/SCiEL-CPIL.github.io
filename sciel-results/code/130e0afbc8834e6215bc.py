import numpy as np
from scipy.spatial import Delaunay

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    eps = 1e-9
    
    # Initialize with inverse distance
    H = 1.0 / (distance_matrix + eps)
    
    # Set diagonal to 0
    np.fill_diagonal(H, 0.0)
    
    try:
        # 1. Reconstruct coordinates via MDS
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (distance_matrix ** 2) @ J
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        
        # Sort eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Take top 2 dimensions
        # Ensure eigenvalues are non-negative for sqrt
        eigenvalues[:2] = np.maximum(eigenvalues[:2], 0)
        points = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
        
        # 2. Delaunay Triangulation
        tri = Delaunay(points)
        delaunay_edges = set()
        for simplex in tri.simplices:
            i, j, k = simplex
            delaunay_edges.add((min(i, j), max(i, j)))
            delaunay_edges.add((min(i, k), max(i, k)))
            delaunay_edges.add((min(j, k), max(j, k)))
            
        # 3. Boost Delaunay edges
        # The optimal TSP tour is a subgraph of the Delaunay triangulation.
        # We multiply the heuristic for these edges by a factor to guide the ants.
        bonus = 3.0
        
        # Efficiently update H
        # Since n is small (50), a loop is fine, but we can also use a mask
        # Let's stick to the loop for clarity and safety with the set lookup
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in delaunay_edges:
                    H[i, j] *= bonus
                    H[j, i] *= bonus
                else:
                    # Penalize non-Delaunay edges to discourage crossing/long edges
                    # A factor of 0.5 might be good
                    H[i, j] *= 0.5
                    H[j, i] *= 0.5
                    
    except Exception:
        # If any error occurs, fall back to the simple inverse distance heuristic
        # H is already initialized with 1/(d+eps)
        pass
        
    # Ensure all values are positive
    H = np.maximum(H, 1e-9)
    
    return H
