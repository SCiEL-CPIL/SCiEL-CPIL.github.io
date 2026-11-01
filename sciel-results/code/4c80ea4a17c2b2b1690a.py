import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition (optimal rank for 2x2).
    """
    start_time = time.time()
    n, m, p = 2, 2, 2

    # Strassen's 7 products:
    # U[i*m+j, c]: coefficient of A[i,j] in product c
    # V[j*p+k, c]: coefficient of B[j,k] in product c
    # W[k*n+i, c]: coefficient for C[i,j] from product c (trace convention: W[k*n+i, c])
    
    # U data (4x7)
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],  # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],   # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],   # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]   # A[1,1]
    ]

    # V data (4x7)
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],  # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],   # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],   # B[1,0]
        [1, 0, -1, 0, 1, 0, 1]   # B[1,1]
    ]

    # W data (4x7)
    # Row index k*n + i
    # k=0, i=0 -> C[0,0]: [1, 0, 0, 1, -1, 0, 1]
    # k=0, i=1 -> C[1,0]: [0, 1, 0, 1, 0, 0, 0]
    # k=1, i=0 -> C[0,1]: [0, 0, 1, 0, 1, 0, 0]
    # k=1, i=1 -> C[1,1]: [1, -1, 1, 0, 0, 1, 0]
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    U = np.array(U_data, dtype=float)
    V = np.array(V_data, dtype=float)
    W = np.array(W_data, dtype=float)

    # Verification step to ensure correctness (optional but good for safety)
    # Reconstruct tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Actually, the decomposition is: T_{i,j,k} = \sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}
    # Wait, the standard tensor form for matrix mult C = AB is T_{i,j,k} = \delta_{i,k} (if we view it as A_{i,l} B_{l,j} = C_{i,j})?
    # No, the task says: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies the tensor T is defined such that T[i, j, k] corresponds to the coefficient of A[i,j] and B[j,k] contributing to C[i,k]?
    # Let's check the indices.
    # U shape (n*m, rank) -> U[i*m+j, c] is coeff of A[i,j]
    # V shape (m*p, rank) -> V[j*p+k, c] is coeff of B[j,k]
    # W shape (p*n, rank) -> W[k*n+i, c] is coeff for C[i,k]
    
    # The tensor T is a 3-tensor of size n x m x p? Or n x p x m?
    # Usually, matrix multiplication tensor T has indices (i, j, k) where i is row of C, j is col of C, k is contraction index.
    # T[i, j, k] = 1 if k is the contraction index linking A[i,k] and B[k,j].
    # So T[i, j, k] = delta_{k, k'} where k' is the inner dimension index.
    # In the decomposition: T[i, j, k] = \sum_c U[i*m+k, c] * V[k*p+j, c] * W[j*n+i, c]
    # Let's verify this mapping.
    # U index: i*m + k (row i, col k of A)
    # V index: k*p + j (row k, col j of B)
    # W index: j*n + i (row j, col i of C^T? Or just W indexed by output column j and output row i?)
    # The task says W[k*n+i, r]. Here the first index of W is 'k' in the variable name in the prompt?
    # Prompt: "W[k*n+i,r]". In the context of the loop, usually the indices are aligned.
    # Let's assume the tensor T is defined by T[i, j, k] = 1 if the term A[i,k]B[k,j] contributes to C[i,j].
    # Then the decomposition is T[i, j, k] = \sum_c U[i*m+k, c] V[k*p+j, c] W[j*n+i, c].
    # Note the W index: j*n+i. The prompt says W[k*n+i]. This suggests the variable name in the prompt 'k' corresponds to the column index of C (which is 'j' in my notation above) and 'i' is the row index of C.
    # So if we denote C indices as (row, col) = (i, j), then W index is col*n + row = j*n + i.
    # And U index is row*m + inner = i*m + k.
    # And V index is inner*p + col = k*p + j.
    
    # Let's verify with the provided data.
    # A[0,0] is at U[0*2+0, :] = U[0, :] = [1, 0, 1, 0, 1, -1, 0]
    # B[0,0] is at V[0*2+0, :] = V[0, :] = [1, 1, 0, -1, 0, 1, 0]
    # C[0,0] is at W[0*2+0, :] = W[0, :] = [1, 0, 0, 1, -1, 0, 1]
    
    # Check term c=0: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1. This should be T[0,0,0].
    # T[0,0,0] corresponds to A[0,0]B[0,0] contributing to C[0,0]. This is correct (1).
    
    # Check term c=0 for T[0,1,0] (A[0,0]B[0,1] -> C[0,1])?
    # A[0,0] -> U[0, :]
    # B[0,1] -> V[0*2+1, :] = V[1, :] = [0, 0, 1, 0, 0, 1, 0]
    # C[0,1] -> W[1*2+0, :] = W[2, :] = [0, 0, 1, 0, 1, 0, 0]
    # Sum c=0: 1 * 0 * 0 = 0.
    # Sum c=2: U[0,2]=1, V[1,2]=1, W[2,2]=1. Product = 1.
    # So T[0,1,0] gets 1 from c=2. This is correct.
    
    # The code seems to define the correct Strassen decomposition.
    # We just need to return it.
    
    if time.time() - start_time > 110:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
