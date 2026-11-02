"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-35 decomposition using 5 blocks of Strassen's rank-7 2x2x2 multiplication.
    The 2x2x10 multiplication tensor is a direct sum of five 2x2x2 blocks.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    
    num_blocks = p // 2  # 5 blocks
    rank_per_block = 7   # Strassen rank for 2x2x2
    rank = num_blocks * rank_per_block  # 35
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen's 7 products for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for b in range(num_blocks):
        base_col = b * rank_per_block
        # Column offsets for the 7 Strassen products
        cols = [base_col + s for s in range(rank_per_block)]
        
        # A indices: i*m + j
        # A00=0, A01=1, A10=2, A11=3
        # B indices: j*p + k, for block b: k in {2b, 2b+1}
        # B00 = 0*p + 2b = 2b
        # B01 = 0*p + (2b+1) = 2b+1
        # B10 = 1*p + 2b = p + 2b
        # B11 = 1*p + (2b+1) = p + 2b + 1
        # C indices: k*n + i
        # C[0, 2b] = 2b*n + 0 = 4b
        # C[1, 2b] = 2b*n + 1 = 4b + 1
        # C[0, 2b+1] = (2b+1)*n + 0 = 4b + 2
        # C[1, 2b+1] = (2b+1)*n + 1 = 4b + 3
        
        B00 = 2 * b
        B01 = 2 * b + 1
        B10 = p + 2 * b
        B11 = p + 2 * b + 1
        
        C00 = 4 * b      # k=2b, i=0
        C01 = 4 * b + 1  # k=2b, i=1
        C10 = 4 * b + 2  # k=2b+1, i=0
        C11 = 4 * b + 3  # k=2b+1, i=1
        
        # M1 = (A00+A11)(B00+B11)
        U[0][cols[0]] += 1   # A00
        U[3][cols[0]] += 1   # A11
        V[B00][cols[0]] += 1
        V[B11][cols[0]] += 1
        W[C00][cols[0]] += 1   # C00
        W[C11][cols[0]] += 1   # C11
        
        # M2 = (A10+A11)B00
        U[2][cols[1]] += 1   # A10
        U[3][cols[1]] += 1   # A11
        V[B00][cols[1]] += 1
        W[C10][cols[1]] += 1   # C10
        W[C11][cols[1]] += -1  # C11
        
        # M3 = A00(B01-B11)
        U[0][cols[2]] += 1   # A00
        V[B01][cols[2]] += 1
        V[B11][cols[2]] += -1
        W[C10][cols[2]] += 1   # C01
        W[C11][cols[2]] += 1   # C11
        
        # M4 = A11(B10-B00)
        U[3][cols[3]] += 1   # A11
        V[B10][cols[3]] += 1
        V[B00][cols[3]] += -1
        W[C00][cols[3]] += 1   # C00
        W[C10][cols[3]] += 1   # C10
        
        # M5 = (A00+A01)B11
        U[0][cols[4]] += 1   # A00
        U[1][cols[4]] += 1   # A01
        V[B11][cols[4]] += 1
        W[C00][cols[4]] += -1  # C00
        W[C10][cols[4]] += 1   # C01
        
        # M6 = (A10-A00)(B00+B10)
        U[2][cols[5]] += 1   # A10
        U[0][cols[5]] += -1  # A00
        V[B00][cols[5]] += 1
        V[B10][cols[5]] += 1
        W[C11][cols[5]] += 1   # C11
        
        # M7 = (A01-A11)(B01+B11)
        U[1][cols[6]] += 1   # A01
        U[3][cols[6]] += -1  # A11
        V[B01][cols[6]] += 1
        V[B11][cols[6]] += 1
        W[C00][cols[6]] += 1   # C00
        W[C11][cols[6]] += 1   # C11
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
