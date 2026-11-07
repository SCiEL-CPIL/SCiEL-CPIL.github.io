import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.linalg import eigh

def heuristics(distance_matrix):
    start_time = time.time()
    n = distance_matrix.shape[0]
    
    # 1. MDS to get 2D coords
    # Classical MDS
    # B = -0.5 * J * D^2 * J
    D2 = distance_matrix ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    # Eigen decomposition
    # We only need the top 2 eigenvalues/vectors
    vals, vecs = eigh(B)
    # Sort by eigenvalue descending
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    # Coordinates are sqrt(eigenvalue) * eigenvector
    coords = vecs[:, :2] * np.sqrt(np.maximum(vals[:2], 0))
    
    # 2. Delaunay Triangulation
    try:
        tri = Delaunay(coords)
        # Get edges
        # tri.simplices contains indices of points in each triangle
        # We need unique edges
        edges = set()
        for simplex in tri.simplices:
            edges.add((simplex[0], simplex[1]))
            edges.add((simplex[1], simplex[2]))
            edges.add((simplex[2], simplex[0]))
        
        # Create a boolean matrix for Delaunay edges
        is_delaunay = np.zeros((n, n), dtype=bool)
        for i, j in edges:
            is_delaunay[i, j] = True
            is_delaunay[j, i] = True
    except Exception:
        # Fallback if Delaunay fails (e.g., collinear points)
        is_delaunay = np.zeros((n, n), dtype=bool)
        # Just use k-NN as fallback for structure
        # But for now, assume it works for random points
    
    # 3. Compute "Local Tangent" / Alignment
    # For each node i, compute the average direction to its Delaunay neighbors
    # This vector represents the "flow" or "tangent" of the local structure
    tangent_vectors = np.zeros((n, 2))
    for i in range(n):
        neighbors = np.where(is_delaunay[i, :])[0]
        if len(neighbors) > 0:
            # Vector from i to each neighbor
            vecs_to_neighbors = coords[neighbors] - coords[i]
            # Normalize and sum? Or just sum?
            # Sum of unit vectors gives a direction
            norms = np.linalg.norm(vecs_to_neighbors, axis=1)
            norms[norms == 0] = 1
            unit_vecs = vecs_to_neighbors / norms[:, np.newaxis]
            tangent_vectors[i] = np.sum(unit_vecs, axis=0)
    
    # 4. Construct Heuristic
    eps = 1e-9
    d = distance_matrix + eps
    
    # Base distance heuristic
    alpha = 2.0 # Tuned
    base_h = d ** (-alpha)
    
    # Alignment score
    # For each edge (i, j), calculate angle between (coords[j] - coords[i]) and tangent_vectors[i]
    # We want to reward edges that align with the local flow
    alignment_scores = np.ones((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v_edge = coords[j] - coords[i]
            norm_edge = np.linalg.norm(v_edge)
            if norm_edge < 1e-9:
                continue
            v_edge_unit = v_edge / norm_edge
            
            v_tan = tangent_vectors[i]
            norm_tan = np.linalg.norm(v_tan)
            if norm_tan < 1e-9:
                # No clear tangent, neutral score
                alignment_scores[i, j] = 1.0
                continue
            
            v_tan_unit = v_tan / norm_tan
            
            # Cosine similarity
            cos_sim = np.dot(v_edge_unit, v_tan_unit)
            # Clamp to [0, 1] to penalize anti-parallel strongly?
            # Or use cos^2 to be symmetric?
            # The directive says "rewarding edges that are ... aligned".
            # Let's use max(0, cos_sim) ** 2 to reward alignment and penalize deviation.
            alignment_scores[i, j] = max(0.0, cos_sim) ** 2
    
    # Combine
    # Boost Delaunay edges significantly
    # h = base_h * (1 + boost * is_delaunay) * alignment_scores
    # Or: h = base_h * alignment_scores * (1 + 5 * is_delaunay)
    
    boost = 5.0
    structure_factor = 1.0 + boost * is_delaunay
    
    h = base_h * alignment_scores * structure_factor
    
    # Ensure positivity and finiteness
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    # Diagonal should be 0 or small? 
    # The problem says "edge-attractiveness matrix". 
    # Ants pick *unvisited* nodes. Diagonal is not picked.
    # But to be safe, set diagonal to 0 or 1e-9.
    np.fill_diagonal(h, 0.0)
    
    # Time check
    if time.time() - start_time > 290:
        # Return a simple fallback if we are running out of time
        # Though this is unlikely
        pass
        
    return h
