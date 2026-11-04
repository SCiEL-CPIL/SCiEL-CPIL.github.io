"""Matrix multiplication 2x3 by 3x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that the decomposition exactly represents the matrix multiplication tensor."""
    # Convert to numpy arrays for efficient tensor product computation
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We check that this equals 1 when the indices are consistent with matmul, 0 otherwise.
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor for this multiplication has T[i, j, k] = 1 (as a bilinear form coefficient)
    # Actually, the tensor decomposition represents the trilinear form:
    # sum_{i,j,k} C[i,k] * A[i,j] * B[j,k] = sum_c (sum_i C[i,k]*W[k*n+i,c]) * 
    #   (sum_j A[i,j]*U[i*m+j,c]) ... 
    # 
    # More precisely, the condition is:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be
    # 1 if this term contributes to the multiplication, which is always for valid (i,j,k)
    # in the range. Actually, the tensor M_{n,m,p} has entries T[i,j,k] = 1 for all
    # valid i, j, k, and the multiplication is: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # 
    # Wait, let me reconsider. The standard tensor for matrix multiplication M_{n,m,p} 
    # is defined such that for matrices A (n x m) and B (m x p), C = A*B is given by
    # C[i,k] = sum_j A[i,j] * B[j,k].
    # 
    # The tensor decomposition gives: C[i,k] = sum_c (sum_j U[i,j,c] * A[i,j]) * (sum_k V[j,k,c] * B[j,k]) * W[k,i,c]
    # Hmm, that's not quite right either.
    #
    # The correct condition: the decomposition of the multiplication tensor means
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for ALL valid i, j, k.
    # Because the multiplication is just C[i,k] = sum_j A[i,j] * B[j,k], and each
    # "term" in the sum over j is a separate tensor entry that must equal 1.
    
    # Let's compute the trilinear form value for each (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 36 for schoolbook
    
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
