import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin for 300s limit

    n = distance_matrix.shape[0]
    if n < 2:
        return np.ones_like(distance_matrix)

    # --- 1. Graph Laplacian Spectral Clustering ---
    # Model the problem as finding natural partitions in the customer graph.
    # High attractiveness for edges within clusters, low for edges crossing cuts.

    # Create a weighted adjacency matrix based on distances.
    # Smaller distance -> higher weight.
    # We use 1/(d + eps) as weight to avoid division by zero.
    eps = 1e-9
    weights = 1.0 / (distance_matrix + eps)
    # Set diagonal to 0
    np.fill_diagonal(weights, 0.0)

    # Build sparse Laplacian
    L = laplacian(csr_matrix(weights))

    # Find the Fiedler vector (2nd smallest eigenvalue's eigenvector)
    # This vector identifies the natural cut in the graph.
    try:
        # k=2 to get the first two eigenvectors (0th is constant, 1st is Fiedler)
        # sigma='LA' means Largest Algebraic? No, we want smallest non-zero.
        # eigsh with sigma=0 finds eigenvalues near 0.
        # Since L is positive semi-definite, the smallest eigenvalue is 0.
        # We want the next one.
        # Using k=2 with sigma='SM' might be tricky if 0 is an eigenvalue.
        # Better: shift-invert or just get k=3 smallest and take the 2nd non-trivial?
        # Actually, for a connected graph, eigenvalue 0 has multiplicity 1.
        # So we want the eigenvalue with index 1 (0-indexed) in the sorted list of eigenvalues.
        # eigsh(k=2, sigma=0) returns the two eigenvalues closest to 0.
        # Usually these are lambda_0 = 0 and lambda_1 > 0.
        
        # Let's use k=3 to be safe and pick the first non-zero one.
        vals, vecs = eigsh(L, k=3, sigma=0, which='LM')
        
        # Sort by eigenvalue
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        # Find the first eigenvector with non-negligible variance (not the constant one)
        fiedler_vec = None
        for i in range(len(vals)):
            v = vecs[:, i]
            if np.std(v) > 1e-6:
                fiedler_vec = v
                break
        
        if fiedler_vec is None:
            # Fallback: use the last one
            fiedler_vec = vecs[:, -1]

    except Exception:
        # Fallback if spectral decomposition fails
        # Use a simple geometric projection
        fiedler_vec = coordinates[:, 0]

    # --- 2. Construct Heuristic Matrix ---
    # The Fiedler vector provides a 1D embedding of the nodes.
    # Nodes with similar Fiedler values are likely in the same "cluster" or region.
    # We want to encourage routing within these regions.
    
    # Compute pairwise similarity based on Fiedler vector.
    # Similarity = 1 / (1 + |f_i - f_j|^2)
    # This creates a smooth kernel where close values in Fiedler space are similar.
    
    fiedler_diff = np.abs(fiedler_vec[:, np.newaxis] - fiedler_vec[np.newaxis, :])
    similarity = 1.0 / (1.0 + fiedler_diff ** 2)
    
    # Normalize similarity to [0, 1] roughly
    sim_min = np.min(similarity)
    sim_max = np.max(similarity)
    if sim_max > sim_min:
        similarity = (similarity - sim_min) / (sim_max - sim_min)
    else:
        similarity = np.ones_like(similarity)

    # Base heuristic is inverse distance
    base_heur = 1.0 / (distance_matrix + eps)
    np.fill_diagonal(base_heur, 0.0)

    # Normalize base heuristic to [0, 1]
    base_max = np.max(base_heur)
    if base_max > 0:
        base_heur = base_heur / base_max

    # Combine:
    # We want high heuristic for:
    # 1. Short distances (base_heur)
    # 2. Edges within the same spectral cluster (similarity)
    
    # A multiplicative combination might be too harsh.
    # Let's use a weighted sum or a product with scaling.
    # h = base_heur * (1 + alpha * similarity)
    # Or h = base_heur + beta * similarity * base_heur
    
    # Let's try: h = base_heur * (0.5 + 0.5 * similarity)
    # This ensures that even if similarity is low, short edges are still preferred,
    # but edges within clusters get a boost.
    
    # Alternative: Use the Fiedler vector to define "zones".
    # Customers with similar Fiedler values should be visited in the same tour.
    # The ACO heuristic should guide ants to stay within a zone until the tour is full or no more nodes fit.
    
    # Let's refine the similarity.
    # If we use k-means on the Fiedler vector, we get discrete clusters.
    # But the continuous kernel is smoother.
    
    # Let's also consider the demand.
    # High demand nodes are "heavy". Maybe they should be clustered more tightly?
    # Not directly used in the heuristic matrix, but the ACO process handles capacity.
    
    # Final Heuristic:
    # h[i, j] = base_heur[i, j] * (1.0 + gamma * similarity[i, j])
    # Where gamma controls the strength of the spectral clustering bias.
    
    gamma = 2.0  # Strength of spectral bias
    
    heuristic = base_heur * (1.0 + gamma * similarity)
    
    # Ensure positive values
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Set diagonal to 0 (or very small) to avoid self-loops, though ACO usually handles this.
    # The task says "edge-attractiveness matrix". Self-loops are not edges in the routing sense.
    # However, the ACO code picks unvisited customers. So diagonal values don't matter much,
    # but let's set them to 0 or 1e-9 to be safe.
    np.fill_diagonal(heuristic, 0.0)
    
    # Check shape
    if heuristic.shape != distance_matrix.shape:
        heuristic = 1.0 / (distance_matrix + eps)
        np.fill_diagonal(heuristic, 0.0)

    return heuristic
