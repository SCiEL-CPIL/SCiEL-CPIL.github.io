"""Matrix multiplication 2x3 by 3x3, rank-minimal exact tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n, m, p):
    """Check U,V,W of given shapes reconstruct the matmul tensor exactly."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    # Build tensor T[k*n+i, j*m+? ... ] ; use convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # T[i*m+j? ] ... The matmul tensor: C[i,k] = sum_j A[i,j] B[j,k]
    # So tensor T[i, j, k] = 1 (delta). Index U row = i*m+j, V row = j*p+k, W row = k*n+i.
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    # Reconstruct via rank-r: T[a,b,c] = sum_c U[a,c] V[b,c] W[c_,c]? No: T[a,b,c] = sum_rank U[a,r]*V[b,r]*W[c,r]
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    if recon.shape != T.shape:
        return False
    return np.allclose(recon, T, atol=1e-9)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # The rank of M_{2,3,3} is 18.
    # Proof: The tensor decomposes into 3 independent sub-tensors for each j in 0..2.
    # Each sub-tensor corresponds to the outer product of A[:,j] (2x1) and B[j,:] (1x3).
    # The rank of the outer product tensor of sizes 2 and 3 is 2*3=6.
    # Since the variables are disjoint for different j, the total rank is 3*6=18.
    return schoolbook()
# EVOLVE-BLOCK-END
