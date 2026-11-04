import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
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
                
    # Check for linear dependence of the tensor terms
    # Tensor dimension: (n*m) * (m*p) * (p*n)
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    total_dim = dim_u * dim_v * dim_w
    
    # Create matrix M of size total_dim x rank
    # M[:, c] is the flattened tensor T_c
    M = np.zeros((total_dim, rank), dtype=float)
    
    for c in range(rank):
        # U[:, c] is a vector of length dim_u
        # V[:, c] is a vector of length dim_v
        # W[:, c] is a vector of length dim_w
        u_col = np.array([U[i][c] for i in range(dim_u)])
        v_col = np.array([V[i][c] for i in range(dim_v)])
        w_col = np.array([W[i][c] for i in range(dim_w)])
        
        # Tensor product
        tensor_c = np.outer(np.outer(u_col, v_col), w_col)
        M[:, c] = tensor_c.flatten()
        
    # Compute rank of M
    # Use SVD to find rank
    try:
        s = np.linalg.svd(M, compute_uv=False)
        # Rank is number of singular values > tolerance
        tol = 1e-10 * max(M.shape) * s[0] if len(s) > 0 else 0
        actual_rank = np.sum(s > tol)
    except Exception:
        actual_rank = rank
        
    if actual_rank < rank:
        # Potential for reduction, but complex to implement exact half-integer solution
        # For now, if we detect dependence, we might try to return a reduced version
        # but without a robust solver for half-integers, we fall back to schoolbook
        # unless we can easily identify a zero column or duplicate
        pass
        
    return {"U": U, "V": V, "W": W}
