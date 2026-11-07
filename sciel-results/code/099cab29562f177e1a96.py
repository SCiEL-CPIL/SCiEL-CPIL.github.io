"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, ConvexHull

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    # Fallback heuristic: standard 1/distance
    fallback = 1.0 / (distance_matrix + 1e-9)
    
    try:
        # 1. Reconstruct 2D points from distance matrix using Classical MDS
        D2 = distance_matrix ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * (J @ D2 @ J)
        
        eigvals, eigvecs = np.linalg.eigh(B)
        
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        top_eigvals = eigvals[:2]
        top_eigvecs = eigvecs[:, :2]
        
        top_eigvals = np.maximum(top_eigvals, 0)
        
        points = top_eigvecs * np.sqrt(top_eigvals)
        
        # 2. Compute Delaunay Triangulation
        tri = Delaunay(points)
        
        # 3. Build base heuristic with smooth distance decay
        # Use a power law that strongly favors short edges
        dist = distance_matrix.copy()
        np.fill_diagonal(dist, 1.0)  # avoid div by zero
        
        # Base heuristic: inverse distance raised to power 1.5
        # This creates a smoother gradient than pure 1/d
        H = 1.0 / (dist ** 1.5 + 1e-9)
        
        # 4. Boost Delaunay edges significantly
        edges = set()
        for simplex in tri.simplices:
            edges.add((simplex[0], simplex[1]))
            edges.add((simplex[1], simplex[2]))
            edges.add((simplex[0], simplex[2]))
            
        for i, j in edges:
            boost = 3.0  # multiply Delaunay edges by 3
            H[i, j] *= boost
            H[j, i] *= boost
        
        # 5. Boost MST edges even more
        # Compute MST using Kruskal's algorithm
        edge_list = []
        for i in range(n):
            for j in range(i+1, n):
                edge_list.append((distance_matrix[i, j], i, j))
        edge_list.sort()
        
        # Union-Find for Kruskal
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
        
        mst_edges = set()
        for w, i, j in edge_list:
            if union(i, j):
                mst_edges.add((min(i,j), max(i,j)))
        
        for i, j in mst_edges:
            boost = 5.0  # multiply MST edges by 5
            H[i, j] *= boost
            H[j, i] *= boost
        
        # 6. Boost convex hull edges
        try:
            hull = ConvexHull(points)
            hull_edges = set()
            for simplex in hull.simplices:
                for i in range(len(simplex)):
                    j = (i + 1) % len(simplex)
                    hull_edges.add((min(simplex[i], simplex[j]), max(simplex[i], simplex[j])))
            
            for i, j in hull_edges:
                boost = 2.0  # multiply hull edges by 2
                H[i, j] *= boost
                H[j, i] *= boost
        except Exception:
            pass  # If hull computation fails, skip this boost
        
        # 7. Penalize very long edges (crossing penalty)
        # Edges that are much longer than average should be penalized
        avg_dist = np.mean(dist[~np.eye(n, dtype=bool)])
        long_edge_threshold = 1.5 * avg_dist
        
        long_mask = dist > long_edge_threshold
        H[long_mask] *= 0.3  # reduce attractiveness of long edges
        
        # Ensure diagonal is 0
        np.fill_diagonal(H, 0)
        
        # Ensure all values are positive
        H = np.maximum(H, 1e-9)
        
        return H
        
    except Exception:
        return fallback
# EVOLVE-BLOCK-END
