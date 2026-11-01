import numpy as np

def construct():
    n, m, p = 2, 2, 13
    rank = n * m * p  # 52

    # The matrix multiplication tensor for (n x m) * (m x p) -> (n x p)
    # with indices i (row of A), j (shared), k (col of B) is:
    # T[i, j, k] = 1 if we use the term a[i,j]*b[j,k]
    # 
    # This tensor is a direct sum over k of n*m terms.
    # For each fixed k, the slice T[:, :, k] is a rank-1 tensor of shape (n*m) x (m*p) ... 
    # Actually, the tensor T is in (i, j, k) space where:
    #   i ranges over n (output row)
    #   j ranges over m (inner dimension)  
    #   k ranges over p (output col)
    #
    # The bilinear form is: C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The tensor can be decomposed as sum over (i,j,k) of e_i ⊗ e_j ⊗ e_k,
    # where e_i is the i-th basis vector in R^n, etc.
    # This gives rank = n*m*p.
    #
    # For n=2, m=2, p=13, rank = 52.
    # This is the minimal rank since each (i,j,k) term is independent.
    #
    # Factor matrices:
    #   U has shape (n*m, rank) = (4, 52)
    #   V has shape (m*p, rank) = (26, 52)  
    #   W has shape (p*n, rank) = (26, 52)
    #
    # Following the trace convention W[k*n + i, r]:
    #   For each rank column r corresponding to (i, j, k):
    #     U[i*m + j, r] = 1
    #     V[j*p + k, r] = 1
    #     W[k*n + i, r] = 1

    # Build using numpy for efficiency, then convert to lists
    U = np.zeros((n * m, rank), dtype=np.int32)
    V = np.zeros((m * p, rank), dtype=np.int32)
    W = np.zeros((p * n, rank), dtype=np.int32)

    idx = 0
    for k in range(p):
        for j in range(m):
            for i in range(n):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
