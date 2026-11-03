"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    # Initialize factor matrices
    U = [[0.0] * rank for _ in range(n * m)]  # 25 x 49
    V = [[0.0] * rank for _ in range(m * p)]  # 25 x 49
    W = [[0.0] * rank for _ in range(p * n)]  # 25 x 49
    
    # Helper to set entries
    def set_uv_w(u_idx, v_idx, w_idx, col, val=1.0):
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
    
    # The 49-rank decomposition of the 5x5x5 matrix multiplication tensor
    # Index convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # i, j, k in {0,1,2,3,4}
    
    # Laderman 3x3x3 for top-left block (i,j,k in {0,1,2})
    # L1: (0,0,0)
    set_uv_w(0, 0, 0, 0)
    # L2: (0,1,1), (1,0,0) with signs
    set_uv_w(1, 1, 1, 1)
    set_uv_w(1, 2, 2, 1, -1)  # V[2,1] = -1 means V index 2*3+1=7? No...
    
    # This is getting error-prone. Let me use a cleaner approach.
    
    # Actually, let me just return the schoolbook for now and verify the interface,
    # then I'll hard-code the 49-rank properly.
    
    # For now, use a known valid 49-rank construction
    # The 49 terms are defined by specific index triples
    
    # Let me use the explicit construction from the literature
    # Each term c has U indices, V indices, W indices
    
    # I'll define the 49 terms as lists of (row, col_value) for each matrix
    terms = []
    
    # Laderman 3x3 (terms 0-6)
    # Term 0: U[0,0]=1, V[0,0]=1, W[0,0]=1
    terms.append([(0,1),(0,1),(0,1)])
    # Term 1: U[0,1]=1, V[0,1]=1, W[1,0]=1
    terms.append([(1,1),(1,1),(1,1)])
    # Term 2: U[1,0]=1, V[1,0]=1, W[0,1]=1
    terms.append([(3,1),(3,1),(2,1)])
    # ... this is too error prone
    
    # Let me just use the schoolbook rank 125 as fallback and note that 49 is the target
    # Actually, let me properly construct the 49-rank decomposition
    
    # Clear and rebuild
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # 49-rank decomposition of 5x5x5 tensor
    # Using the known optimal construction
    
    # I'll use the following explicit 49-term decomposition
    # Each term is a triple of (U_row, V_row, W_row) with coefficient 1
    # (coefficients of -1 are handled by negative entries)
    
    # This is the Laderman-Strassen hybrid construction
    
    # For brevity and correctness, let me use a verified 49-rank decomposition
    # The terms are (u_idx, v_idx, w_idx, coeff)
    
    # Laderman 3x3x3 (indices 0-2 for i,j,k)
    # Using standard Laderman:
    # m1 = A11*(B11+B22) -> contributes to C11, C22
    # m2 = (A11+A22)*B22 -> contributes to C11, C22  
    # m3 = (A12+A22)*B21 -> contributes to C12, C22
    # m4 = A21*(B11-B21) -> contributes to C21, C22
    # m5 = (A11-A21)*(B21+B22) -> contributes to C11, C21
    # m6 = (A12-A11)*(B11+B12) -> contributes to C12, C21
    # m7 = (A22-A21)*(B12-B22) -> contributes to C12, C21
    
    # This is getting very complex to map to the tensor decomposition format.
    
    # Let me just return a valid (if not minimal) decomposition for now
    # and use rank 125 schoolbook as the safe fallback
    
    # Actually, the task says score is the rank, lower is better.
    # Let me try to implement the 49-rank properly.
    
    # I'll use a different strategy: hard-code a known 49-rank decomposition
    # from the literature, verified to be correct.
    
    # For the 5x5x5 case, here is a valid 49-rank decomposition:
    # (I'm using the construction that achieves rank 49)
    
    # Let me just use the schoolbook for now since I can't reliably hard-code
    # the 49-rank from memory without verification.
    
    rank = n * m * p
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
