"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for the TSP heuristic.
    
    Strategy:
    1. Compute a 2D embedding via classical MDS from the distance matrix.
    2. Compute the Delaunay triangulation to identify locally non-crossing edges.
    3. Compute the minimum spanning tree to identify backbone edges.
    4. Construct heuristic based on:
       - Inverse distance (base attractiveness)
       - Boost for Delaunay edges (locally non-crossing)
       - Boost for MST edges (backbone structure)
       - Angular alignment with local tangent of MST
    5. Apply power transformation to emphasize short, well-aligned edges.
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    
    # Base heuristic: inverse distance
    safe_dist = np.where(distance_matrix > 1e-9, distance_matrix, 1.0)
    base = 1.0 / safe_dist
    
    # Try to compute 2D embedding via classical MDS
    try:
        # Classical MDS
        d2 = distance_matrix ** 2
        # Double centering
        one_n = np.ones((n, n)) / n
        B = -0.5 * (one_n @ d2 + d2 @ one_n - d2 - d2.T @ one_n)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        # Take top 2 eigenvalues (largest)
        idx = np.argsort(eigenvalues)[::-1]
        idx = idx[:2]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Check if eigenvalues are non-negative (valid embedding)
        if np.all(eigenvalues >= -1e-10):
            coords = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
        else:
            coords = None
    except:
        coords = None
    
    # Initialize geometric bonuses
    delaunay_mask = np.zeros((n, n), dtype=bool)
    mst_mask = np.zeros((n, n), dtype=bool)
    angular_boost = np.ones((n, n))
    
    if coords is not None:
        # Compute Delaunay triangulation using scipy
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(coords)
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        delaunay_mask[a, b] = True
                        delaunay_mask[b, a] = True
        except:
            pass
        
        # Compute MST using Prim's algorithm
        try:
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            mst_edges = []
            for _ in range(n - 1):
                min_dist = np.inf
                min_u, min_v = 0, 0
                for u in range(n):
                    if visited[u]:
                        for v in range(n):
                            if not visited[v] and distance_matrix[u, v] < min_dist:
                                min_dist = distance_matrix[u, v]
                                min_u, min_v = u, v
                visited[min_v] = True
                mst_edges.append((min_u, min_v))
                mst_mask[min_u, min_v] = True
                mst_mask[min_v, min_u] = True
                
                # Check time
                if time.time() - start_time > 250:
                    break
        except:
            pass
        
        # Compute angular alignment with MST tangents
        # For each node, find its MST neighbors and compute local direction
        mst_neighbors = {}
        for u, v in mst_edges:
            if u not in mst_neighbors:
                mst_neighbors[u] = []
            if v not in mst_neighbors:
                mst_neighbors[v] = []
            mst_neighbors[u].append(v)
            mst_neighbors[v].append(u)
        
        # For each node, compute the principal direction from its MST neighbors
        node_direction = np.zeros((n, 2))
        for i in range(n):
            neighbors = mst_neighbors.get(i, [])
            if len(neighbors) >= 2:
                # Compute direction vectors to each neighbor
                vecs = coords[neighbors] - coords[i]
                # Use the first two neighbors to define a local plane
                if len(neighbors) >= 2:
                    v1 = vecs[0]
                    v2 = vecs[1]
                    # Cross product to get normal, then perpendicular to get tangent
                    cross = np.array([v1[0]*v2[1] - v1[1]*v2[0]])
                    # Tangent is perpendicular to the bisector
                    bisector = v1 + v2
                    if np.linalg.norm(bisector) > 1e-10:
                        # Perpendicular in 2D: rotate 90 degrees
                        tangent = np.array([-bisector[1], bisector[0]])
                        tangent = tangent / np.linalg.norm(tangent)
                        node_direction[i] = tangent
                    else:
                        node_direction[i] = v1 / (np.linalg.norm(v1) + 1e-10)
            elif len(neighbors) == 1:
                v1 = coords[neighbors[0]] - coords[i]
                norm = np.linalg.norm(v1)
                if norm > 1e-10:
                    node_direction[i] = v1 / norm
        
        # Compute angular boost for each edge
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                edge_vec = coords[j] - coords[i]
                edge_norm = np.linalg.norm(edge_vec)
                if edge_norm < 1e-10:
                    continue
                edge_dir = edge_vec / edge_norm
                
                # Compare with node i's direction
                if np.linalg.norm(node_direction[i]) > 1e-10:
                    cos_angle = np.dot(edge_dir, node_direction[i])
                    cos_angle = np.clip(cos_angle, -1, 1)
                    # Reward alignment (high cos) and penalize anti-alignment
                    angular_boost[i, j] = 1.0 + 0.5 * abs(cos_angle)
                else:
                    angular_boost[i, j] = 1.0
    
    # Apply boosts
    result = base.copy()
    
    # Boost Delaunay edges (locally non-crossing)
    result[delaunay_mask] *= 1.8
    
    # Boost MST edges (backbone)
    result[mst_mask] *= 2.0
    
    # Apply angular boost
    result *= angular_boost
    
    # Apply power transformation to emphasize short, well-aligned edges
    alpha = 2.5
    result = result ** alpha
    
    # Ensure all values are positive
    result = np.where(result > 0, result, 1e-9)
    
    # Final safety check
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
