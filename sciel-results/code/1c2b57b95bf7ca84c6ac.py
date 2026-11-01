"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] equals 1 iff i==k, else 0."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Expected: T[i,j,k] = 1 if i == k, else 0
    # Reshape W to (p, n, rank) so W[k, i, c]
    W_reshaped = W.T  # (rank, p*n) -> (p*n, rank) is W, so W.T is (rank, p*n)
    # Let's do it step by step
    # U: (n*m, rank) -> (n, m, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # First: A[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c]
    # = np.einsum('ijc,jkc->ijk', U_r, V_r)
    A = np.einsum('ijc,jkc->ijk', U_r, V_r)
    # Then T[i,j,k] = sum_c ... but we already summed over c in A.
    # Actually, we need: T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # = sum_c A[i,j,k,c] * W_r[k,i,c] but A doesn't have c index.
    # Let's do it in two steps:
    # B[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c]  -> this is A
    # T[i,j,k] = sum_c ... wait, the sum over c is already in A.
    # No, the correct formula is:
    # T[i,j,k] = sum_c (U_r[i,j,c] * V_r[j,k,c]) * W_r[k,i,c]
    # But U_r[i,j,c] * V_r[j,k,c] summed over c gives A[i,j,k].
    # Then T[i,j,k] = A[i,j,k] * W_r[k,i, ?] -- no, this is wrong.
    
    # Correct approach:
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    # = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    # Expected: T[i,j,k] = 1 if i == k, else 0
    expected = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i == k:
                    expected[i, j, k] = 1.0
    
    return np.allclose(T, expected, atol=1e-10)

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Schoolbook decomposition
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Verify
    assert _verify_decomposition(U, V, W, n, m, p), "Schoolbook decomposition failed verification"
    
    elapsed = time.time() - t0
    if elapsed > 140:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
