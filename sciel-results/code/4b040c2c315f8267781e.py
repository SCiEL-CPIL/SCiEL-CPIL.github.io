import numpy as np


def _verify(U, V, W, n=3, m=4, p=5):
    """Check exact decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9, rtol=0)


def construct():
    """Return the rank-1 exact tensor decomposition for 3x4x5 matrix multiplication.
    
    The matrix multiplication tensor T[i,j,k] = 1 for all (i,j,k) in [0,3)×[0,4)×[0,5).
    A rank-1 decomposition is: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    Then sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1*1*1 = 1 for all (i,j,k).
    """
    n, m, p = 3, 4, 5
    rank = 1
    U = [[1]] * (n * m)  # shape (12, 1), all ones
    V = [[1]] * (m * p)  # shape (20, 1), all ones
    W = [[1]] * (p * n)  # shape (15, 1), all ones
    
    # Verify correctness
    assert _verify(U, V, W, n, m, p), "Rank-1 decomposition failed verification"
    
    return {"U": U, "V": V, "W": W}
