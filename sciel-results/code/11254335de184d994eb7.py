"""Program using spectral gap heuristic for Maximum Independent Set via partitioning."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def priority(adjacency):
    """
    Assign scores to vertices using a spectral gap heuristic based on graph partitioning.
    
    The idea:
    1. Compute the graph Laplacian L = D - A.
    2. Find the Fiedler vector (eigenvector of the second-smallest eigenvalue).
    3. Use the Fiedler vector to partition vertices into two sets: those with positive
       Fiedler values (side A) and those with negative Fiedler values (side B).
    4. Identify the "smaller" partition (fewer vertices) and prioritize vertices in that
       partition, as selecting from the smaller side minimizes cross-boundary conflicts.
    5. Within each partition, further prioritize lower-degree vertices to reduce local conflicts.
    6. The score combines: partition membership (prefer smaller side), degree penalty,
       and a spectral coherence term (how well-aligned a vertex is with its partition).
    
    Returns:
        A numpy array of scores, one per vertex. Higher score = selected earlier.
    """
    n = adjacency.shape[0]
    A = np.asarray(adjacency, dtype=float)
    
    # Ensure symmetry
    A = (A + A.T) / 2.0
    
    # Compute degree vector
    degrees = A.sum(axis=1)
    
    # Handle case where graph is empty or trivial
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([1.0])
    
    # Build Laplacian
    D = np.diag(degrees)
    L = D - A
    
    # For efficiency with larger graphs, use sparse representation
    L_sparse = sp.csr_matrix(L)
    
    try:
        # Find the Fiedler vector (second-smallest eigenvalue's eigenvector)
        k = min(2, n)
        if n > 1:
            vals, vecs = spla.eigsh(L_sparse, k=k, sigma=0, which='LM')
        else:
            vals = np.array([0.0])
            vecs = np.array([[1.0]])
        
        # Sort by eigenvalue
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        # Fiedler vector is the eigenvector corresponding to the second-smallest eigenvalue
        if len(vals) >= 2:
            fiedler = vecs[:, 1]
        else:
            # Fallback: use degree-based scoring
            fiedler = np.zeros(n)
        
        # Normalize Fiedler vector to have zero mean and unit variance
        fiedler_mean = np.mean(fiedler)
        fiedler_std = np.std(fiedler)
        if fiedler_std > 1e-10:
            fiedler = (fiedler - fiedler_mean) / fiedler_std
        else:
            fiedler = np.zeros(n)
        
        # Partition vertices based on sign of Fiedler values
        # Positive Fiedler values -> side A, Negative -> side B
        side_A_mask = fiedler >= 0
        side_B_mask = ~side_A_mask
        
        n_A = np.sum(side_A_mask)
        n_B = np.sum(side_B_mask)
        
        # The smaller partition is more valuable to prioritize
        # We want to select from the smaller side first
        if n_A <= n_B:
            # Side A is smaller or equal, prioritize Side A
            partition_score = np.where(side_A_mask, 1.0, 0.0)
            smaller_side_size = n_A
        else:
            # Side B is smaller, prioritize Side B
            partition_score = np.where(side_B_mask, 1.0, 0.0)
            smaller_side_size = n_B
        
        # Spectral coherence: how strongly a vertex aligns with its partition
        # Vertices with extreme Fiedler values are more "committed" to their side
        # Use |fiedler| as a measure of commitment
        coherence = np.abs(fiedler)
        
        # Normalize coherence to [0, 1]
        max_coherence = np.max(coherence) if np.max(coherence) > 0 else 1.0
        coherence_normalized = coherence / max_coherence
        
        # Degree penalty: lower degree is better
        max_degree = np.max(degrees) if np.max(degrees) > 0 else 1.0
        degree_normalized = degrees / max_degree
        
        # Combine scores:
        # - partition_score: 1.0 if in smaller partition, 0.0 otherwise (strong signal)
        # - coherence_normalized: reward vertices strongly aligned with their partition
        # - degree_normalized: penalize high-degree vertices
        #
        # Weighting: Partition membership should dominate, but coherence and degree
        # provide fine-grained ordering within each partition.
        alpha_partition = 2.0  # Strong weight for being in the smaller partition
        alpha_coherence = 0.5  # Moderate weight for spectral coherence
        alpha_degree = 1.0     # Full weight for degree penalty
        
        scores = (alpha_partition * partition_score + 
                  alpha_coherence * coherence_normalized - 
                  alpha_degree * degree_normalized)
        
        # Ensure scores are well-behaved
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        
    except Exception:
        # Fallback to simple degree-based scoring if spectral computation fails
        scores = -degrees.astype(float) / (np.max(degrees) + 1e-10)
    
    return scores
# EVOLVE-BLOCK-END
