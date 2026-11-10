# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates using a spectral optimization approach on low-weight vectors.
    
    Strategy:
    1. Generate a pool of integer vectors with weights 1, 2, and 3 (entries in {-1, 0, 1}).
    2. Build a compatibility graph where an edge exists if the squared distance between two vectors is >= the maximum of their squared norms.
    3. Use the principal eigenvector of the adjacency matrix to rank nodes.
    4. Greedily select nodes in descending order of spectral centrality to form a clique.
    
    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    d = 11
    
    # 1. Generate pool of low-weight vectors
    candidates = []
    
    # Weight 1
    for i in range(d):
        for sign in [-1, 1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = sign
            candidates.append(v)
            
    # Weight 2
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
                    
    # Weight 3
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        for sk in [-1, 1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
                            
    n_candidates = len(candidates)
    candidates = np.array(candidates, dtype=np.int64)
    
    # 2. Build compatibility matrix
    # dist_sq[i, j] = ||v_i - v_j||^2
    # norm_sq[i] = ||v_i||^2
    # Condition: dist_sq[i, j] >= max(norm_sq[i], norm_sq[j])
    
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Compute squared distances
    # (a-b)^2 = a^2 + b^2 - 2ab
    # dist_sq[i,j] = norms_sq[i] + norms_sq[j] - 2 * dot(v_i, v_j)
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # Max norm squared for each pair
    max_norms_sq = np.maximum(norms_sq[:, None], norms_sq[None, :])
    
    # Compatibility matrix (True if compatible)
    # Diagonal is False (distinct points)
    compatible = dist_sq >= max_norms_sq
    np.fill_diagonal(compatible, False)
    
    # 3. Spectral Optimization
    # Convert boolean to float for eigen decomposition
    A = compatible.astype(np.float64)
    
    try:
        # Compute eigenvalues and eigenvectors
        # We only need the principal eigenvector
        # For large N, this can be expensive, but N~1500 is manageable
        vals, vecs = np.linalg.eigh(A)
        # Principal eigenvector is the one corresponding to the largest eigenvalue
        # eigh returns sorted ascending, so last one is largest
        principal_vec = vecs[:, -1]
    except Exception:
        # Fallback to degree if eigen fails
        principal_vec = np.sum(compatible, axis=1).astype(np.float64)
        
    # Rank candidates by magnitude of principal eigenvector component
    # Higher magnitude indicates higher centrality in the clique structure
    order = np.argsort(-np.abs(principal_vec))
    
    # 4. Greedy Clique Construction
    selected_indices = []
    selected_vectors = []
    
    for idx in order:
        # Check if candidate is compatible with all selected
        is_compatible = True
        for sel_idx in selected_indices:
            if not compatible[idx, sel_idx]:
                is_compatible = False
                break
        
        if is_compatible:
            selected_indices.append(idx)
            selected_vectors.append(candidates[idx])
            
    # Convert to numpy array
    if not selected_vectors:
        # Fallback: return a single point or empty? Task requires k>=2.
        # If greedy failed, try natural order
        selected_vectors = [candidates[0], candidates[1]]
        
    points = np.array(selected_vectors, dtype=np.int64)
    
    # Final validation and rounding (though ints are exact)
    points = np.around(points).astype(np.int64)
    
    return points

# EVOLVE-BLOCK-END
