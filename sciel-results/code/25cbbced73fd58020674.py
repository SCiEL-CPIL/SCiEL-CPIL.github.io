"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank calculated:
    # Part 1 (A11 * B1): 3 * Strassen(7) + Schoolbook(4) = 25
    # Part 2 (A12 * B2): 7 * 2 = 14
    # Part 3 (A21 * B1): 7 * 2 = 14
    # Part 4 (A22 * B2): 7 * 1 = 7
    # Total: 60
    rank = 60
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val

    # Part 1: A11 (2x2) * B1 (2x7)
    # B1 consists of columns k=0..6, rows j=0,1
    # We process k in blocks of 2 for Strassen, and last one separately
    
    # Strassen blocks for k in [0,1], [2,3], [4,5]
    for block_start in [0, 2, 4]:
        k0 = block_start
        k1 = block_start + 1
        
        # M1 = (A00 + A11) * (B00 + B11)
        # A00: U[0], A11: U[3]
        # B00: V[0*7+k0]=V[k0], B11: V[1*7+k0]=V[7+k0]
        # Wait, B1 is 2x7.
        # B1[j, k]. V index is j*7 + k.
        # For Strassen, we need B00, B01, B10, B11 for the 2x2 block.
        # Here the 2x2 block is B1[:, k0:k1+1].
        # B00 = B1[0, k0] -> V[0*7 + k0]
        # B01 = B1[0, k1] -> V[0*7 + k1]
        # B10 = B1[1, k0] -> V[1*7 + k0]
        # B11 = B1[1, k1] -> V[1*7 + k1]
        
        # M1
        add_term(0, 0, 0) # Placeholder, need to sum U and V
        # Actually, add_term sets single indices. Strassen requires sums.
        # I must manually construct the U and V vectors for each of the 7 terms.
        
        # Let's define the 7 terms for one Strassen block
        # Term 1: (A00 + A11) * (B00 + B11)
        # U: A00 (idx 0) + A11 (idx 3)
        # V: B00 (idx k0) + B11 (idx 7+k0) ... wait, B00 is B[0,k0], B11 is B[1,k1]?
        # No, Strassen on A (2x2) and B (2x2).
        # B in this block is columns k0, k1.
        # B_block = [[B1[0,k0], B1[0,k1]], [B1[1,k0], B1[1,k1]]]
        # So B00=B1[0,k0], B01=B1[0,k1], B10=B1[1,k0], B11=B1[1,k1].
        
        # Term 1: (A00+A11)(B00+B11)
        U[0][col] = 1; U[3][col] = 1
        V[k0][col] = 1; V[7+k1][col] = 1 # B00 is V[0*7+k0], B11 is V[1*7+k1]
        W[k0*3+0][col] = 1 # C00 part?
        # Strassen output:
        # C00 = M1 - M4 + M6
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M7
        
        # We need to accumulate W contributions.
        # It is easier to just write the 7 terms explicitly with their W coefficients.
        
        # Term 1: M1
        U[0][col] = 1; U[3][col] = 1
        V[k0][col] = 1; V[7+k1][col] = 1
        W[k0*3+0][col] = 1  # +1 to C00
        W[k1*3+1][col] = 1  # +1 to C11
        col += 1
        
        # Term 2: M2 = (A10 + A11) * B00
        U[1][col] = 1; U[3][col] = 1 # A10 is idx 1, A11 is idx 3
        V[k0][col] = 1 # B00
        W[k0*3+1][col] = 1 # +1 to C10
        col += 1
        
        # Term 3: M3 = A00 * (B01 + B11)
        U[0][col] = 1
        V[7+k1][col] = 1; V[k1][col] = 1 # B11 is V[7+k1], B01 is V[k1]
        W[k1*3+0][col] = 1 # +1 to C01
        W[k1*3+1][col] = -1 # -1 to C11
        col += 1
        
        # Term 4: M4 = A11 * (B10 + B00)
        U[3][col] = 1
        V[7+k0][col] = 1; V[k0][col] = 1 # B10 is V[7+k0], B00 is V[k0]
        W[k0*3+0][col] = -1 # -1 to C00
        W[k0*3+1][col] = 1 # +1 to C10
        col += 1
        
        # Term 5: M5 = (A00 + A01) * B11
        U[0][col] = 1; U[2][col] = 1 # A00 idx 0, A01 idx 2
        V[7+k1][col] = 1 # B11
        W[k1*3+0][col] = 1 # +1 to C01
        col += 1
        
        # Term 6: M6 = (A10 - A00) * (B00 + B01)
        U[1][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[k1][col] = 1
        W[k0*3+0][col] = 1 # +1 to C00
        col += 1
        
        # Term 7: M7 = (A01 - A11) * (B10 + B11)
        U[2][col] = 1; U[3][col] = -1
        V[7+k0][col] = 1; V[7+k1][col] = 1
        W[k1*3+1][col] = 1 # +1 to C11
        col += 1

    # Last column k=6 for Part 1 (Schoolbook 4 terms)
    k6 = 6
    # C06 = A00 B06 + A01 B16
    # C16 = A10 B06 + A11 B16
    
    # Term 8: A00 * B06
    U[0][col] = 1
    V[k6][col] = 1 # B[0,6]
    W[k6*3+0][col] = 1
    col += 1
    
    # Term 9: A01 * B16
    U[2][col] = 1
    V[7+k6][col] = 1 # B[1,6]
    W[k6*3+0][col] = 1
    col += 1
    
    # Term 10: A10 * B06
    U[1][col] = 1
    V[k6][col] = 1
    W[k6*3+1][col] = 1
    col += 1
    
    # Term 11: A11 * B16
    U[3][col] = 1
    V[7+k6][col] = 1
    W[k6*3+1][col] = 1
    col += 1

    # Part 2: A12 (2x1) * B2 (1x7)
    # A12: i in {0,1}, j=2. U indices: 2, 5.
    # B2: j=2, k in 0..6. V indices: 14+k.
    # C: i in {0,1}, k in 0..6. W indices: k*3 + i.
    
    for k in range(7):
        # C0k += A12_0 * B2_k
        U[2][col] = 1
        V[14+k][col] = 1
        W[k*3+0][col] = 1
        col += 1
        
        # C1k += A12_1 * B2_k
        U[5][col] = 1
        V[14+k][col] = 1
        W[k*3+1][col] = 1
        col += 1

    # Part 3: A21 (1x2) * B1 (2x7)
    # A21: i=2, j in {0,1}. U indices: 6, 7.
    # B1: j in {0,1}, k in 0..6. V indices: 0..13.
    # C: i=2, k in 0..6. W indices: k*3 + 2.
    
    for k in range(7):
        # C2k += A21_0 * B1_0k
        U[6][col] = 1
        V[k][col] = 1
        W[k*3+2][col] = 1
        col += 1
        
        # C2k += A21_1 * B1_1k
        U[7][col] = 1
        V[7+k][col] = 1
        W[k*3+2][col] = 1
        col += 1

    # Part 4: A22 (1x1) * B2 (1x7)
    # A22: i=2, j=2. U index: 8.
    # B2: j=2, k in 0..6. V indices: 14+k.
    # C: i=2, k in 0..6. W indices: k*3 + 2.
    
    for k in range(7):
        U[8][col] = 1
        V[14+k][col] = 1
        W[k*3+2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
