"""Matrix multiplication 3x4x6 - schoolbook rank 72 decomposition."""

def construct():
    n, m, p = 3, 4, 6
    
    # Schoolbook decomposition
    # Rank = n * m * p = 3 * 4 * 6 = 72
    rank = n * m * p
    
    # U shape: (n*m, rank) = (12, 72)
    # V shape: (m*p, rank) = (24, 72)
    # W shape: (p*n, rank) = (18, 72)
    # Note: W index convention is W[k*n + i, r] for C[i, k]
    # Standard convention for matrix multiplication tensor:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T_{i, j, k} = 1 if we use A[i,j] and B[j,k].
    # U corresponds to A indices (i, j) -> index i*m + j
    # V corresponds to B indices (j, k) -> index j*p + k
    # W corresponds to C indices (i, k) -> index k*n + i (per prompt: W[k*n+i,r])
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index for A[i, j]
                u_idx = i * m + j
                # V index for B[j, k]
                v_idx = j * p + k
                # W index for C[i, k]
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
