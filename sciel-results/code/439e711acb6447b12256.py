"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition.
    # 5 blocks of 2 columns (using Strassen rank 7)
    # 1 block of 1 column (using trivial rank 4)
    # Total rank = 5 * 7 + 4 = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a Strassen block for columns k, k+1
    def add_strassen_block(k1, k2, start_idx):
        # k1, k2 are column indices in B (and C)
        # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B indices: j*p + k
        # C indices: k*n + i
        
        # P1: (A00+A11)(B0_k1 + B1_k2)
        # A00 is idx 0, A11 is idx 3
        # B0_k1 is idx 0*p + k1, B1_k2 is idx 1*p + k2
        
        # We need to add 7 terms.
        # Term 1
        U[0][start_idx] += 1
        U[3][start_idx] += 1
        V[0*p + k1][start_idx] += 1
        V[1*p + k2][start_idx] += 1
        # C contributions:
        # C00 = P1 ...
        # C00 is idx k1*n + 0
        # C11 is idx k2*n + 1
        # Wait, C indices are k*n + i.
        # C_{0, k1} is idx k1*n + 0
        # C_{1, k2} is idx k2*n + 1
        # C_{0, k2} is idx k2*n + 0
        # C_{1, k1} is idx k1*n + 1
        
        # Strassen:
        # C00 = P1 - P5 + P7
        # C01 = P3 + P5  (Here 01 means row 0, col 1 of the block? No, block is cols k1, k2. So C_{0,k2} = P3+P5)
        # C10 = P2 + P4  (C_{1,k1} = P2+P4)
        # C11 = P1 - P2 + P6 (C_{1,k2} = P1-P2+P6)
        
        # Let's map carefully.
        # Block A is 2x2. Block B is 2x2 (cols k1, k2). Block C is 2x2 (cols k1, k2).
        # C[0, k1] = P1 - P5 + P7
        # C[0, k2] = P3 + P5
        # C[1, k1] = P2 + P4
        # C[1, k2] = P1 - P2 + P6
        
        # Term 1: P1
        # U: A00, A11
        # V: B0_k1, B1_k2
        # W: C0_k1 (+1), C1_k2 (+1)
        W[k1*n + 0][start_idx] += 1
        W[k2*n + 1][start_idx] += 1
        col_offset_local = start_idx + 1
        
        # Term 2: P2 = (A10+A11)B0_k1
        U[2][col_offset_local] += 1 # A10
        U[3][col_offset_local] += 1 # A11
        V[0*p + k1][col_offset_local] += 1 # B0_k1
        # W: C1_k1 (+1)
        W[k1*n + 1][col_offset_local] += 1
        col_offset_local += 1
        
        # Term 3: P3 = A00(B0_k2 - B1_k2)
        U[0][col_offset_local] += 1 # A00
        V[0*p + k2][col_offset_local] += 1 # B0_k2
        V[1*p + k2][col_offset_local] -= 1 # B1_k2
        # W: C0_k2 (+1)
        W[k2*n + 0][col_offset_local] += 1
        col_offset_local += 1
        
        # Term 4: P4 = A11(B1_k1 - B0_k1)
        U[3][col_offset_local] += 1 # A11
        V[1*p + k1][col_offset_local] += 1 # B1_k1
        V[0*p + k1][col_offset_local] -= 1 # B0_k1
        # W: C1_k1 (+1)
        W[k1*n + 1][col_offset_local] += 1
        col_offset_local += 1
        
        # Term 5: P5 = (A00+A01)B1_k2
        U[0][col_offset_local] += 1 # A00
        U[1][col_offset_local] += 1 # A01
        V[1*p + k2][col_offset_local] += 1 # B1_k2
        # W: C0_k1 (-1), C0_k2 (+1)
        W[k1*n + 0][col_offset_local] -= 1
        W[k2*n + 0][col_offset_local] += 1
        col_offset_local += 1
        
        # Term 6: P6 = (A10-A00)(B0_k1 + B0_k2)
        U[2][col_offset_local] += 1 # A10
        U[0][col_offset_local] -= 1 # A00
        V[0*p + k1][col_offset_local] += 1 # B0_k1
        V[0*p + k2][col_offset_local] += 1 # B0_k2
        # W: C1_k2 (+1)
        W[k2*n + 1][col_offset_local] += 1
        col_offset_local += 1
        
        # Term 7: P7 = (A01-A11)(B1_k1 + B1_k2)
        U[1][col_offset_local] += 1 # A01
        U[3][col_offset_local] -= 1 # A11
        V[1*p + k1][col_offset_local] += 1 # B1_k1
        V[1*p + k2][col_offset_local] += 1 # B1_k2
        # W: C0_k1 (+1)
        W[k1*n + 0][col_offset_local] += 1
        
    # Add 5 Strassen blocks
    for i in range(5):
        k1 = 2*i
        k2 = 2*i + 1
        add_strassen_block(k1, k2, col_offset)
        col_offset += 7
        
    # Add 1 trivial block for col 10
    # Rank 4 for M_2,2,1
    # C0_k = A00 B0_k + A01 B1_k
    # C1_k = A10 B0_k + A11 B1_k
    
    # Term 8: A00 B0_10
    U[0][col_offset] += 1
    V[0*p + 10][col_offset] += 1
    W[10*n + 0][col_offset] += 1
    col_offset += 1
    
    # Term 9: A01 B1_10
    U[1][col_offset] += 1
    V[1*p + 10][col_offset] += 1
    W[10*n + 0][col_offset] += 1
    col_offset += 1
    
    # Term 10: A10 B0_10
    U[2][col_offset] += 1
    V[0*p + 10][col_offset] += 1
    W[10*n + 1][col_offset] += 1
    col_offset += 1
    
    # Term 11: A11 B1_10
    U[3][col_offset] += 1
    V[1*p + 10][col_offset] += 1
    W[10*n + 1][col_offset] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
