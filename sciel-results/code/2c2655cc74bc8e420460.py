"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reconstructs the matrix multiplication tensor.
    
    The reconstruction is:
    (A * B)_{i, k} = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For this to equal sum_j A_{i,j} B_{j,k} for all A, B, we need:
    sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j, j'}
    for all i, j, j', k.
    """
    rank = U.shape[1]
    
    # Efficient check: for random A, B, verify the decomposition matches direct multiplication
    rng = np.random.RandomState(42)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    AB_direct = A @ B
    
    # U[i*m+j, r] -> (U_reshaped)_{i,j,r}
    U_reshaped = U.T.reshape(rank, n, m)
    # V[j*p+k, r] -> (V_reshaped)_{j,k,r}
    V_reshaped = V.T.reshape(rank, m, p)
    # W[k*n+i, r] -> (W_reshaped)_{i,k,r}
    W_reshaped_temp = W.T.reshape(rank, p, n)
    W_reshaped = W_reshaped_temp.transpose(0, 2, 1)
    
    # tr(U_r^T A) = sum_{i,j} U_r[i,j] * A[i,j]
    U_A = np.einsum('rij,ij->r', U_reshaped, A)
    V_B = np.einsum('rjk,jk->r', V_reshaped, B)
    
    # sum_r U_A[r] * V_B[r] * W_r
    AB_decomp = np.einsum('r,r,rik->ik', U_A, V_B, W_reshaped)
    
    return np.allclose(AB_direct, AB_decomp, atol=1e-10)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    U, V, W = _schoolbook(n, m, p)
    
    # Verify the schoolbook decomposition works
    if not _verify(U, V, W, n, m, p):
        # Should never happen, but just in case
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # The schoolbook rank is n*m*p = 75
    # We want to try to reduce the rank.
    
    # For now, let's just return the schoolbook decomposition as a baseline.
    # The key challenge is finding a provably correct lower-rank decomposition.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
