"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 11
    # Use Strassen for the first 10 columns (5 blocks of 2), schoolbook for the last column.
    # Each Strassen block of size 2x2x2 uses rank 7.
    # Schoolbook for 2x2x1 uses rank 4.
    # Total rank = 5 * 7 + 4 = 39.
    
    U = [[0.0] * 39 for _ in range(n * m)]
    V = [[0.0] * 39 for _ in range(m * p)]
    W = [[0.0] * 39 for _ in range(p * n)]
    
    col_offset = 0
    
    # Strassen blocks for columns 0-9 (5 blocks)
    for block in range(5):
        k0 = block * 2
        k1 = k0 + 1
        base = col_offset
        
        # P1 = (A00 + A11)(B0k0 + B1k1)
        # U: A00, A11 -> rows 0, 3
        # V: B0k0, B1k1 -> rows 0*k0, 1*k1 (m*p indices: j*p+k)
        # W: C0k0, C1k1 -> rows k0*n+0, k1*n+1
        U[0][base+0] += 1; U[3][base+0] += 1
        V[0*p+k0][base+0] += 1; V[1*p+k1][base+0] += 1
        W[k0*n+0][base+0] += 1; W[k1*n+1][base+0] += 1
        
        # P2 = (A10 + A11)B0k1
        U[1][base+1] += 1; U[3][base+1] += 1
        V[0*p+k1][base+1] += 1
        W[k1*n+1][base+1] += 1
        
        # P3 = A00(B1k1 - B0k0)
        U[0][base+2] += 1
        V[1*p+k1][base+2] += 1; V[0*p+k0][base+2] -= 1
        W[k1*n+0][base+2] += 1; W[k1*n+1][base+2] += 1
        
        # P4 = A10(B0k0 - B0k1)
        U[1][base+3] += 1
        V[0*p+k0][base+3] += 1; V[0*p+k1][base+3] -= 1
        W[k0*n+1][base+3] += 1; W[k1*n+1][base+3] += 1
        
        # P5 = (A00 + A01)B1k0
        U[0][base+4] += 1; U[2][base+4] += 1
        V[1*p+k0][base+4] += 1
        W[k0*n+0][base+4] += 1; W[k1*n+0][base+4] += 1
        
        # P6 = (A11 - A00)(B0k1 + B1k1)
        U[3][base+5] += 1; U[0][base+5] -= 1
        V[0*p+k1][base+5] += 1; V[1*p+k1][base+5] += 1
        W[k1*n+1][base+5] += 1
        
        # P7 = (A01 - A10)(B0k0 + B1k0)
        U[2][base+6] += 1; U[1][base+6] -= 1
        V[0*p+k0][base+6] += 1; V[1*p+k0][base+6] += 1
        W[k0*n+0][base+6] += 1
        
        col_offset += 7
    
    # Schoolbook for the last column (k=10)
    # C0,10 = A00*B0,10 + A01*B1,10
    # C1,10 = A10*B0,10 + A11*B1,10
    base = col_offset
    k_last = 10
    
    # Term 1: A00 * B0,k_last
    U[0][base+0] += 1
    V[0*p+k_last][base+0] += 1
    W[k_last*n+0][base+0] += 1
    
    # Term 2: A01 * B1,k_last
    U[2][base+1] += 1
    V[1*p+k_last][base+1] += 1
    W[k_last*n+0][base+1] += 1
    
    # Term 3: A10 * B0,k_last
    U[1][base+2] += 1
    V[0*p+k_last][base+2] += 1
    W[k_last*n+1][base+2] += 1
    
    # Term 4: A11 * B1,k_last
    U[3][base+3] += 1
    V[1*p+k_last][base+3] += 1
    W[k_last*n+1][base+3] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
