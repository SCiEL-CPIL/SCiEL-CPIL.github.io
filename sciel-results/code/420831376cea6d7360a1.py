"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
import time


def priority(adjacency):
    """
    Assign a score to each vertex such that a greedy decoder picking vertices
    in descending score order (skipping neighbors of already-picked vertices)
    yields a large independent set.
    
    Strategy:
    1. Convert adjacency to sparse matrix.
    2. Compute degree.
    3. Use spectral methods on the graph Laplacian to identify vertices that
       are 'peripheral' or in low-density regions, which tend to be better
       choices for independent set.
    4. Combine degree-based and spectral scores.
    """
    start_time = time.time()
    deadline = start_time + 100  # 100 second deadline with safety margin
    
    n = adjacency.shape[0]
    
    # Compute degree
    degree = adjacency.sum(axis=1).astype(float)
    
    # Build sparse Laplacian
    A = csr_matrix(adjacency.astype(np.float64))
    L = laplacian(A, normed=False)
    
    # Compute Fiedler vector (second smallest eigenvector) using power iteration
    # on the shifted Laplacian to get the eigenvector corresponding to the
    # second smallest eigenvalue
    # We use the fact that the smallest eigenvalue is 0 with eigenvector 1/n
    # So we can find the second smallest by deflation
    
    # Use scipy's sparse eigensolver if available, otherwise fall back to simple heuristic
    try:
        from scipy.sparse.linalg import eigsh
        
        # Find the smallest non-zero eigenvalue's eigenvector
        # Use shift-invert mode for better accuracy
        k = min(3, n - 1)
        if k > 1:
            # Find smallest eigenvalues
            vals, vecs = eigsh(L, k=k, sigma=0, which='LM')
            # The first eigenvector should be constant (eigenvalue 0)
            # The second is the Fiedler vector
            if k >= 2:
                fiedler = vecs[:, 1]
            else:
                fiedler = vecs[:, 0]
        else:
            fiedler = np.ones(n)
        
        # The Fiedler vector gives a spectral embedding
        # Vertices with values close to the mean are more 'central'
        # Vertices far from the mean are more 'peripheral'
        
        # For independent set, we want to pick vertices that are peripheral
        # and have low degree
        
        fiedler_mean = np.mean(fiedler)
        fiedler_std = np.std(fiedler)
        
        if fiedler_std > 1e-10:
            fiedler_norm = (fiedler - fiedler_mean) / fiedler_std
        else:
            fiedler_norm = np.zeros(n)
        
        # Score: combine low degree (good) and high |fiedler_norm| (peripheral, good)
        # Normalize degree to [0, 1]
        if degree.max() > 0:
            degree_norm = degree / degree.max()
        else:
            degree_norm = np.zeros(n)
        
        # score = -degree_norm + alpha * |fiedler_norm|
        # We want high score for low degree and high peripheral-ness
        alpha = 1.0
        score = -degree_norm + alpha * np.abs(fiedler_norm)
        
    except Exception:
        # Fallback: just use negative degree
        score = -degree
    
    # Check deadline
    if time.time() > deadline:
        return score
    
    return score
# EVOLVE-BLOCK-END
