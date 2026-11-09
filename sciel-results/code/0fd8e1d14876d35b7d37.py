"""Rewritten program using spectral partition heuristic for Maximum Independent Set."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def priority(adjacency):
    """
    Assign scores to vertices using a spectral partition heuristic.
    
    The idea:
    1. Compute the graph Laplacian L = D - A.
    2. Find the Fiedler vector (eigenvector of the second-smallest eigenvalue).
    3. The Fiedler vector suggests a bipartition of the graph into V+ and V-.
    4. We identify the "smaller partition" (fewer vertices).
    5. We hypothesize that prioritizing vertices in the smaller partition allows us to
       capture a "core" independent set on that side, while minimizing the number of
       vertices removed from the larger partition due to cross-edges.
    6. Score = -degree + bonus if in smaller partition.
    """
    n = adjacency.shape[0]
    A = np.asarray(adjacency, dtype=float)
    
    # Ensure symmetry
    A = (A + A.T) / 2.0
    
    # Compute degree vector
    degrees = A.sum(axis=1)
    
    if n == 0:
        return np.array([], dtype=float)
    
    # Build Laplacian
    D = np.diag(degrees)
    L = D - A
    
    # Sparse representation
    L_sparse = sp.csr_matrix(L)
    
    try:
        # Find the second-smallest eigenvalue and its eigenvector
        # sigma=0 finds eigenvalues near 0
        # We need k=2 to get the 2nd smallest (Fiedler)
        if n > 1:
            # eigsh with sigma=0 and which='LM' finds eigenvalues closest to 0
            # The smallest is 0 (if connected). The next is Fiedler.
            vals, vecs = spla.eigsh(L_sparse, k=2, sigma=0, which='LM')
        else:
            vals = np.array([0.0])
            vecs = np.array([[1.0]])
        
        # Sort eigenvalues
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        # Fiedler vector is the 2nd smallest
        if len(vals) >= 2:
            fiedler = vecs[:, 1]
        else:
            fiedler = np.zeros(n)
            
        # Determine partitions
        # V+ : fiedler >= 0
        # V- : fiedler < 0
        mask_pos = fiedler >= 0
        mask_neg = ~mask_pos
        
        n_pos = np.sum(mask_pos)
        n_neg = np.sum(mask_neg)
        
        # Identify smaller partition
        if n_pos <= n_neg:
            in_small = mask_pos
        else:
            in_small = mask_neg
            
        # Calculate scores
        # Base score: negative degree (lower degree is better)
        # Bonus: large positive value for vertices in the smaller partition
        # We scale the bonus to be significant relative to degree differences
        # Typical degree in ER(200, p) is ~ 200p. 
        # Let's set bonus to be roughly the average degree to make it a strong preference
        avg_deg = np.mean(degrees) if np.any(degrees > 0) else 1.0
        bonus = avg_deg * 0.5 # Moderate bonus
        
        scores = -degrees + bonus * in_small
        
        # Add a tiny random tie-breaker? No, deterministic is preferred for reproducibility, 
        # but slight noise can help if the heuristic is flat. 
        # However, the task says "fixed greedy decoder". 
        # If scores are identical, the order is arbitrary (likely index order).
        # To avoid bias towards low indices, we can add a very small random noise?
        # The prompt implies a deterministic function. I will stick to deterministic.
        
    except Exception:
        # Fallback
        scores = -degrees.astype(float)
        
    return scores
# EVOLVE-BLOCK-END
