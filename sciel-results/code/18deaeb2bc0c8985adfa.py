"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    # Rank-18 decomposition using Strassen for 2x2x2 blocks and schoolbook for 2x2x1
    # n=2, m=2, p=5
    # U: (n*m, rank) = (4, 18)
    # V: (m*p, rank) = (10, 18)
    # W: (p*n, rank) = (10, 18)
    
    n, m, p = DIMENSIONS
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Index conventions:
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    # We have 5 columns of B (k=0..4)
    # Block 1: k=0,1 -> Strassen (7 terms)
    # Block 2: k=2,3 -> Strassen (7 terms)
    # Block 3: k=4   -> Schoolbook (4 terms)
    # Total: 18 terms
    
    col = 0
    
    # Strassen terms for a 2x2x2 block:
    # M1 = A[0,0] * B[0,0]
    # M2 = A[0,0] * B[1,0]
    # M3 = A[1,0] * B[0,1]
    # M4 = A[1,0] * B[1,1]
    # M5 = A[0,1] * (B[1,0] - B[0,0])
    # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[1,0])
    # M7 = (A[0,1] - A[1,1]) * (B[1,1] + B[1,0])  -- wait, let me use standard Strassen
    
    # Standard Strassen for C = A*B where A is 2x2, B is 2x2, C is 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each 2-column block of B, we use 7 Strassen terms.
    # The A matrix is the same 2x2 input matrix.
    # B columns are different for each block.
    
    # Block 1: B columns k=0,1
    # Block 2: B columns k=2,3
    # Block 3: B column k=4 (schoolbook)
    
    def add_strassen_block(k_offset, col_start):
        """Add 7 Strassen terms for B columns k_offset and k_offset+1."""
        nonlocal col
        # k0 = k_offset, k1 = k_offset+1
        k0 = k_offset
        k1 = k_offset + 1
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00 (i=0,j=0) and A11 (i=1,j=1)
        # V: B00 (j=0,k=k0) and B11 (j=1,k=k1)
        # W: C00 (k=k0,i=0), C01 (k=k1,i=0), C10 (k=k0,i=1), C11 (k=k1,i=1)
        # C00 += M1, C01 -= M5... wait let me re-derive
        
        # Actually let me just carefully map each term.
        
        # Term 1: M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        # U[0*2+0] = U[0] += 1, U[1*2+1] = U[3] += 1
        # V[0*p+k0] = V[k0] += 1, V[1*p+k1] = V[p+k1] += 1
        # W[k0*n+0] = W[k0] += 1 (C00)
        # W[k1*n+0] = W[k1] += 1 (C01)
        # W[k0*n+1] = W[n+k0] += 1 (C10)
        # W[k1*n+1] = W[n+k1] += 1 (C11)
        r = col_start + 0
        U[0][r] += 1
        U[3][r] += 1
        V[k0][r] += 1
        V[p + k1][r] += 1
        W[k0][r] += 1
        W[k1][r] += 1
        W[n + k0][r] += 1
        W[n + k1][r] += 1
        col = r + 1
        
        # Term 2: M2 = (A[1,0] + A[1,1]) * B[0,0]
        # U[1*2+0]=U[2] += 1, U[1*2+1]=U[3] += 1
        # V[0*p+k0]=V[k0] += 1
        # W[k0*n+1]=W[n+k0] += 1 (C10)
        # W[k1*n+1]=W[n+k1] -= 1 (C11)  [from C11 = M1 - M2 + M3 + M6]
        r = col
        U[2][r] += 1
        U[3][r] += 1
        V[k0][r] += 1
        W[n + k0][r] += 1
        W[n + k1][r] -= 1
        col = r + 1
        
        # Term 3: M3 = A[0,0] * (B[1,0] - B[1,1])
        # U[0*2+0]=U[0] += 1
        # V[1*p+k0]=V[p+k0] += 1, V[1*p+k1]=V[p+k1] -= 1
        # W[k0*n+0]=W[k0] -= 1 (C00)
        # W[k1*n+0]=W[k1] -= 1 (C01)
        # W[k0*n+1]=W[n+k0] += 1 (C10)
        # W[k1*n+1]=W[n+k1] += 1 (C11)
        r = col
        U[0][r] += 1
        V[p + k0][r] += 1
        V[p + k1][r] -= 1
        W[k0][r] -= 1
        W[k1][r] -= 1
        W[n + k0][r] += 1
        W[n + k1][r] += 1
        col = r + 1
        
        # Term 4: M4 = A[1,1] * (B[0,1] - B[0,0])
        # U[1*2+1]=U[3] += 1
        # V[0*p+k1]=V[k1] += 1, V[0*p+k0]=V[k0] -= 1
        # W[k0*n+0]=W[k0] += 1 (C00)
        # W[k1*n+0]=W[k1] += 1 (C01)
        # W[k1*n+1]=W[n+k1] += 1 (C11)
        r = col
        U[3][r] += 1
        V[k1][r] += 1
        V[k0][r] -= 1
        W[k0][r] += 1
        W[k1][r] += 1
        W[n + k1][r] += 1
        col = r + 1
        
        # Term 5: M5 = (A[0,0] + A[0,1]) * B[1,1]
        # U[0*2+0]=U[0] += 1, U[0*2+1]=U[1] += 1
        # V[1*p+k1]=V[p+k1] += 1
        # W[k0*n+0]=W[k0] -= 1 (C00)
        # W[k1*n+0]=W[k1] += 1 (C01)
        r = col
        U[0][r] += 1
        U[1][r] += 1
        V[p + k1][r] += 1
        W[k0][r] -= 1
        W[k1][r] += 1
        col = r + 1
        
        # Term 6: M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        # U[1*2+0]=U[2] += 1, U[0*2+0]=U[0] -= 1
        # V[0*p+k0]=V[k0] += 1, V[0*p+k1]=V[k1] += 1
        # W[k1*n+1]=W[n+k1] += 1 (C11)
        r = col
        U[2][r] += 1
        U[0][r] -= 1
        V[k0][r] += 1
        V[k1][r] += 1
        W[n + k1][r] += 1
        col = r + 1
        
        # Term 7: M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        # U[0*2+1]=U[1] += 1, U[1*2+1]=U[3] -= 1
        # V[1*p+k0]=V[p+k0] += 1, V[1*p+k1]=V[p+k1] += 1
        # W[k0*n+0]=W[k0] += 1 (C00)
        # W[k1*n+0]=W[k1] -= 1 (C01)
        r = col
        U[1][r] += 1
        U[3][r] -= 1
        V[p + k0][r] += 1
        V[p + k1][r] += 1
        W[k0][r] += 1
        W[k1][r] -= 1
        col = r + 1
        
        return col
    
    # Block 1: k=0,1
    col = add_strassen_block(0, 0)
    # Block 2: k=2,3
    col = add_strassen_block(2, col)
    
    # Block 3: k=4, schoolbook (4 terms)
    # C00 += A00 * B04
    # C01 += A01 * B14  -- wait, C[i,k] = sum_j A[i,j]*B[j,k]
    # For k=4: C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4]
    #           C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4]
    # 4 terms:
    # Term 8: A[0,0] * B[0,4] -> C[0,4]
    # Term 9: A[0,1] * B[1,4] -> C[0,4]
    # Term 10: A[1,0] * B[0,4] -> C[1,4]
    # Term 11: A[1,1] * B[1,4] -> C[1,4]
    
    k4 = 4
    r = col
    U[0][r] = 1
    V[k4][r] = 1
    W[k4][r] = 1  # C[0,4] = W[4*2+0] = W[8]
    col = r + 1
    
    r = col
    U[1][r] = 1
    V[p + k4][r] = 1
    W[k4][r] = 1  # C[0,4]
    col = r + 1
    
    r = col
    U[2][r] = 1
    V[k4][r] = 1
    W[n + k4][r] = 1  # C[1,4] = W[4*2+1] = W[9]
    col = r + 1
    
    r = col
    U[3][r] = 1
    V[p + k4][r] = 1
    W[n + k4][r] = 1  # C[1,4]
    col = r + 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
