"""Matrix multiplication 2x2 by 2x6 - Lazard-Teitelbaum style bilinear circuit synthesis."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook: rank = n*m*p = 24
    # We construct the rank-24 decomposition corresponding to the standard
    # matrix multiplication algorithm.
    
    # The tensor T for matrix multiplication M_{n,m,p} is defined such that
    # C[i,k] = sum_j A[i,j] B[j,k].
    # In the trace convention, the tensor components are T[i,j,k] = 1.
    # The decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # For the schoolbook algorithm, each term corresponds to a single scalar product A[i,j] * B[j,k].
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U for A[i,j]
                u_idx = i * m + j
                # Index in V for B[j,k]
                v_idx = j * p + k
                # Index in W for C[i,k] (using trace convention W[k*n+i])
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
