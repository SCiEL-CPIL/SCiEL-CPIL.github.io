"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 120.0  # seconds
_SAFETY_MARGIN = 5.0  # seconds


def _is_deadline_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def _fiedler_vector(adjacency, n):
    """
    Compute the Fiedler vector (eigenvector of the second-smallest eigenvalue
    of the graph Laplacian). Returns an array of length n with the Fiedler
    values, or None on failure.
    """
    if n < 3:
        # For very small graphs, just return zeros
        return np.zeros(n)
    
    # Build Laplacian: L = D - A
    degree = adjacency.sum(axis=1)
    # L = D - A
    # For sparse eigensolver, we need L as a sparse matrix
    # L = diag(degree) - A
    # We can construct it as:
    #   L = -A + diag(degree)
    
    # Convert adjacency to float for sparse operations
    A = csr_matrix(adjacency.astype(np.float64))
    
    # Laplacian L = D - A
    # D is diagonal with degree values
    from scipy.sparse import diags
    D = diags(degree.astype(np.float64))
    L = D - A
    
    # We want the second-smallest eigenvalue's eigenvector.
    # eigsh with which='SM' finds smallest magnitude, but we want smallest algebraic.
    # For Laplacian, eigenvalues are non-negative, so smallest magnitude = smallest algebraic.
    # But we want the SECOND smallest. 
    # Strategy: find the two smallest eigenvalues/eigenvectors and take the second one.
    # Or: shift the matrix so that the second smallest becomes the smallest in magnitude.
    
    # Simpler approach: use eigsh with which='SM' and k=2 to get the two smallest.
    # The first will be the zero eigenvalue (constant vector), the second is Fiedler.
    
    try:
        k = min(2, n - 1)
        if k < 2:
            # Not enough eigenvalues to get a meaningful Fiedler vector
            return np.zeros(n)
        
        # For sparse matrices, eigsh with which='SM' can be slow/unstable.
        # Alternative: use which='SA' (smallest algebraic) but scipy's eigsh 
        # supports 'SA' in newer versions. Let's try 'SM' first.
        
        # Actually, for Laplacian, the smallest eigenvalue is 0. 
        # We can use the fact that the Fiedler vector is orthogonal to the all-ones vector.
        # Another approach: compute eigenvectors for the two smallest eigenvalues.
        
        # Let's try using eigsh with which='SM' and k=2
        vals, vecs = eigsh(L, k=k, which='SM')
        
        # Sort by eigenvalue
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        if len(vals) >= 2:
            fiedler = vecs[:, 1]
            # Normalize
            norm = np.linalg.norm(fiedler)
            if norm > 1e-12:
                fiedler = fiedler / norm
            return fiedler
        else:
            return np.zeros(n)
            
    except Exception:
        return None


def priority(adjacency):
    """
    Compute priority scores for each vertex using the Fiedler vector of the
    graph Laplacian. Vertices with larger absolute Fiedler values are prioritized,
    as they lie at natural graph cuts/periphery and are less entangled in dense cores.
    """
    n = adjacency.shape[0]
    adjacency = np.asarray(adjacency, dtype=bool)
    
    # Compute degree-based score as fallback
    degree = adjacency.sum(axis=1).astype(float)
    base_score = -degree  # Lower degree = higher priority
    
    # Try to compute Fiedler vector
    fiedler = _fiedler_vector(adjacency, n)
    
    if fiedler is None:
        # Fallback to degree-based
        return base_score
    
    # Check deadline
    if _is_deadline_exceeded():
        return base_score
    
    # Use absolute Fiedler values as priority
    # Vertices with larger |fiedler[i]| are more "peripheral" or at cut boundaries
    fiedler_score = np.abs(fiedler)
    
    # Combine: primarily use Fiedler magnitude, with a small degree-based tiebreaker
    # Normalize both to [0, 1] range for combination
    f_min, f_max = fiedler_score.min(), fiedler_score.max()
    if f_max - f_min > 1e-12:
        fiedler_norm = (fiedler_score - f_min) / (f_max - f_min)
    else:
        fiedler_norm = np.zeros(n)
    
    d_min, d_max = degree.min(), degree.max()
    if d_max - d_min > 1e-12:
        degree_norm = (degree - d_min) / (d_max - d_min)
    else:
        degree_norm = np.zeros(n)
    
    # Weighted combination: 80% Fiedler, 20% inverse degree
    combined = 0.8 * fiedler_norm - 0.2 * degree_norm
    
    return combined
# EVOLVE-BLOCK-END
