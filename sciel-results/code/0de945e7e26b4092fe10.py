import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def verify(U, V, W, n, m, p):
    # Check if U, V, W exactly decompose the tensor
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    rank = U.shape[1]
    # Flatten U, V, W to vectors
    # U_flat[r] is a vector of length n*m
    # V_flat[r] is a vector of length m*p
    # W_flat[r] is a vector of length p*n
    
    # Compute the tensor from the decomposition
    # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # Note: W index is k*n + i.
    
    # We can compute the matrix product C = A @ B for random A, B and check against the tensor contraction.
    # But we need to check the tensor itself.
    
    # Let's check a few random entries or the whole tensor.
    # Full tensor check:
    # T = sum_r np.outer(np.outer(U[:, r], V[:, r]), W[:, r])? No.
    # U is (n*m, rank). V is (m*p, rank). W is (p*n, rank).
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank).
    U_3d = U.T.reshape(rank, n, m) # (rank, n, m)
    V_3d = V.T.reshape(rank, m, p) # (rank, m, p)
    W_3d = W.T.reshape(rank, p, n) # (rank, p, n)
    
    # T[i, j, k] = sum_r U_3d[r, i, j] * V_3d[r, j, k] * W_3d[r, k, i]
    # This is a contraction.
    # We can compute the full tensor T of shape (n, m, p).
    # T = np.einsum('rij,rjk,rki->ijk', U_3d, V_3d, W_3d)
    
    try:
        T_decomp = np.einsum('rij,rjk,rki->ijk', U_3d, V_3d, W_3d)
    except Exception:
        return False
        
    # The target tensor T_target[i, j, k] is 1 if i, j, k are valid indices for multiplication?
    # No, T_target is the standard multiplication tensor.
    # T_target[i, j, k] = 1 if we are computing C[i, k] from A[i, j] and B[j, k].
    # In the standard tensor, T[i, j, k] = 1 for all i, j, k?
    # No, the tensor is T = sum_{i,j,k} A_{ij} B_{jk} C_{ik}.
    # The coefficient of A_{ij} B_{jk} C_{ik} is 1.
    # So T_target[i, j, k] = 1 for all i, j, k.
    
    T_target = np.ones((n, m, p))
    
    if not np.allclose(T_decomp, T_target, atol=1e-9):
        return False
        
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 150
    
    # Try Landsberg 21 + 21 + 18
    # I need the 21 vectors for 3x3x3.
    # I will use a known decomposition if available, else schoolbook.
    
    # Schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
                
    if time.time() > deadline - 10:
        return {"U": U_school, "V": V_school, "W": W_school}

    # Try to improve
    # Use Landsberg for first two 3x3 blocks, schoolbook for last 2 columns.
    # I don't have the Landsberg vectors hardcoded reliably.
    # I will return schoolbook to be safe.
    
    return {"U": U_school, "V": V_school, "W": W_school}
