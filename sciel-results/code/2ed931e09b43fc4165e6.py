import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 115.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    """
    Returns the standard schoolbook decomposition for matrix multiplication.
    Rank is n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
                
    return U, V, W

def _verify(n, m, p, U, V, W):
    """
    Verifies if the given factor matrices exactly decompose the matrix multiplication tensor.
    T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    For matrix multiplication, T_{i,j,k} should be 1 for all valid indices.
    """
    # Convert to numpy arrays for vectorized verification if inputs are lists
    if isinstance(U, list):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
    else:
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)

    # Construct the full tensor from factors
    # T_reconstructed[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    try:
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    except Exception:
        return False
        
    # The true tensor T_true has T_true[i*m+j, j*p+k, k*n+i] = 1, and 0 elsewhere.
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Check for exact match (with tolerance for floating point)
    if not bool(np.allclose(T_recon, T_true, atol=1e-9)):
        return False
        
    return True

def construct():
    """
    Returns the factor matrices for the matrix multiplication tensor 2x5x5.
    Currently uses the standard schoolbook decomposition with rank 50.
    Includes a verification step to ensure correctness.
    """
    n, m, p = DIMENSIONS
    
    # 1. Generate Schoolbook Decomposition
    U, V, W = _schoolbook(n, m, p)
    
    # 2. Verify
    if not _verify(n, m, p, U, V, W):
        # This should never happen for schoolbook, but good practice
        raise ValueError("Schoolbook decomposition failed verification")
        
    # 3. Return as lists of lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    # No print to avoid overhead during eval
    pass
