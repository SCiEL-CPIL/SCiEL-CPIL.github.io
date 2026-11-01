"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition of rank 7 for 2x2 matrix multiplication
    n, m, p = DIMENSIONS
    rank = 7
    
    # U shape (n*m, rank) = (4, 7)
    # V shape (m*p, rank) = (4, 7)
    # W shape (p*n, rank) = (4, 7)
    
    # Indexing:
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    # r=0: M1 = (A00+A11)(B00+B11) -> C00, C11
    u0 = [1, 0, 0, 1]
    v0 = [1, 0, 0, 1]
    w0 = [1, 0, 0, 1]
    
    # r=1: M2 = (A10+A11)B00 -> C10 (+1), C11 (-1)
    u1 = [0, 0, 1, 1]
    v1 = [1, 0, 0, 0]
    w1 = [0, 1, 0, -1]
    
    # r=2: M3 = A00(B01-B11) -> C01 (+1), C11 (+1)
    u2 = [1, 0, 0, 0]
    v2 = [0, 1, 0, -1]
    w2 = [0, 0, 1, 1]
    
    # r=3: M4 = A11(B10-B00) -> C00 (+1), C10 (+1)
    u3 = [0, 0, 0, 1]
    v3 = [-1, 0, 1, 0]
    w3 = [1, 1, 0, 0]
    
    # r=4: M5 = (A00+A01)B11 -> C00 (-1), C01 (+1)
    u4 = [1, 1, 0, 0]
    v4 = [0, 0, 0, 1]
    w4 = [-1, 0, 1, 0]
    
    # r=5: M6 = (A10-A00)(B00+B01) -> C11 (+1)
    u5 = [-1, 0, 1, 0]
    v5 = [1, 1, 0, 0]
    w5 = [0, 0, 0, 1]
    
    # r=6: M7 = (A01-A11)(B10+B11) -> C00 (+1)
    u6 = [0, 1, 0, -1]
    v6 = [0, 0, 1, 1]
    w6 = [1, 0, 0, 0]
    
    U = [[u0[r], u1[r], u2[r], u3[r], u4[r], u5[r], u6[r]] for r in range(4)]
    V = [[v0[r], v1[r], v2[r], v3[r], v4[r], v5[r], v6[r]] for r in range(4)]
    W = [[w0[r], w1[r], w2[r], w3[r], w4[r], w5[r], w6[r]] for r in range(4)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
