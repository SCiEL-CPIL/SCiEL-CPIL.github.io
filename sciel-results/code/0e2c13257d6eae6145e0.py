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
    3. The Fiedler vector identifies the sparsest cut, dividing the graph into two partitions.
    4. We prioritize vertices in the smaller partition. This is because:
       a. The smaller partition has fewer vertices, so we can potentially select a larger fraction of it.
       b. The cut minimizes edges between partitions, so selecting vertices from one side
          incurs fewer penalties from neighbors in the other side (though greedy decoder
          handles this locally, the global structure helps).
       c. Within the chosen partition, we still prefer lower degree vertices.
    5. Score = Bias * (is_in_smaller_partition) - degree.
    
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
        return np.array([0.0])
    
    # Build Laplacian
    D = np.diag(degrees)
    L = D - A
    
    # For efficiency with larger graphs, use sparse representation
    L_sparse = sp.csr_matrix(L)
    
    try:
        # Find eigenvalues near 0 (smallest ones)
        # We need the second-smallest eigenvalue (Fiedler value)
        # For connected graphs, lambda_0 = 0, lambda_1 > 0.
        # For disconnected graphs, lambda_1 = 0.
        k = min(2, n)
        if n > 1:
            # sigma=0 finds eigenvalues closest to 0
            vals, vecs = spla.eigsh(L_sparse, k=k, sigma=0, which='LM')
            
            # Sort by eigenvalue
            idx = np.argsort(vals)
            vals = vals[idx]
            vecs = vecs[:, idx]
            
            # Check if graph is effectively disconnected (second eigenvalue very close to 0)
            # If so, the Fiedler vector might not provide a good global partition.
            # We check if the second smallest eigenvalue is > 1e-10.
            if len(vals) >= 2 and vals[1] > 1e-10:
                fiedler = vecs[:, 1]
            else:
                # Fallback: Use degree-based scoring if graph is disconnected or ill-conditioned
                # In this case, spectral partitioning is ambiguous.
                # We can still try to use the vector, but it might correspond to components.
                # Let's stick to degree for safety if disconnected.
                fiedler = None
        else:
            fiedler = None
            
        if fiedler is not None:
            # Normalize Fiedler vector
            fiedler_mean = np.mean(fiedler)
            fiedler_std = np.std(fiedler)
            if fiedler_std > 1e-10:
                fiedler = (fiedler - fiedler_mean) / fiedler_std
            else:
                fiedler = np.zeros(n)
                
            # Partition vertices based on sign of Fiedler vector
            # Handle zeros by assigning them to the smaller group arbitrarily or ignoring?
            # Usually few zeros. Let's say > 0 is group 1, <= 0 is group 2.
            mask_1 = fiedler > 0
            mask_2 = ~mask_1
            
            size_1 = np.sum(mask_1)
            size_2 = np.sum(mask_2)
            
            # Determine the smaller partition
            if size_1 <= size_2:
                small_mask = mask_1
            else:
                small_mask = mask_2
                
            # We want to prioritize vertices in the smaller partition.
            # The bias should be strong enough to override degree differences.
            # Max degree is roughly n*p. For n=200, p=0.1, max deg ~ 50.
            # A bias of 100 ensures all vertices in small partition are picked before large partition.
            bias = 100.0
            
            # Score = bias * small_mask - degree
            # Note: degrees are floats.
            scores = bias * small_mask.astype(float) - degrees
            
        else:
            # Fallback to simple degree-based scoring
            scores = -degrees.astype(float)
            
    except Exception:
        # Fallback to simple degree-based scoring if spectral computation fails
        scores = -degrees.astype(float)
    
    return scores
# EVOLVE-BLOCK-END
