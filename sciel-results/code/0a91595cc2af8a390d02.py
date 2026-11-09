"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of T_{2,3,3} as a direct sum of 3 copies of T_{2,3,1}.
    # T_{2,3,1} corresponds to multiplying a 2x3 matrix by a 3x1 vector.
    # The rank of T_{n,m,1} is n*m. Here n=2, m=3, so rank 6 per column.
    # Total rank = 3 * 6 = 18.
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    # Iterate over each column of B (which corresponds to each column of C)
    for k in range(p):
        # For this column k, we perform matrix-vector multiplication A * b_k
        # where b_k is the k-th column of B.
        # The output is the k-th column of C.
        # The tensor slice is T_{n,m,1} = sum_{i=1..n} sum_{j=1..m} e_i \otimes e_{ij} \otimes e_1
        # In the global indices:
        # U index: i*m + j
        # V index: j*p + k  (Note: V index is j*p+k in schoolbook, let's verify)
        # Schoolbook: U[i*m+j], V[j*p+k], W[k*n+i]
        # Let's stick to the schoolbook index mapping to ensure correctness, 
        # but generate it via the block logic.
        
        for i in range(n):
            for j in range(m):
                # The term is A[i,j] * B[j,k] contributing to C[i,k]
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # Assign to the current column of the factor matrices
                U[u_idx][col_idx] = 1
                V[v_idx][col_idx] = 1
                W[w_idx][col_idx] = 1
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
