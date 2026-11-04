"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will use a hybrid decomposition:
    # For i in {0, 1}, we partition the j and k indices into 2x2 blocks.
    # This gives 2 (j-pairs) * 4 (k-pairs) = 8 blocks of size 2x2x2.
    # Each 2x2x2 block uses Strassen's algorithm (rank 7).
    # For i = 2, we use schoolbook (rank 1*4*8 = 32).
    # Total rank = 8 * 7 + 32 = 88.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U coefficients (A)
    U_strassen = [
        [1, 0, 0, 1],   # A00, A01, A10, A11 -> M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    
    # V coefficients (B)
    V_strassen = [
        [1, 0, 1, 0],   # B00, B01, B10, B11 -> M1
        [1, 0, 0, 0],   # M2
        [0, -1, 1, 0],  # M3
        [-1, 1, 0, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    
    # W coefficients (C)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ]
    
    def add_strassen_block(i_start, j_start, k_start):
        nonlocal col
        # Local indices: i' in {0,1}, j' in {0,1}, k' in {0,1}
        # Global indices:
        # U row: (i_start + i') * m + (j_start + j')
        # V row: (j_start + j') * p + (k_start + k')
        # W row: (k_start + k') * n + (i_start + i')
        
        for t in range(7):
            # Update U
            for li in range(2):
                for lj in range(2):
                    val = U_strassen[t][li * 2 + lj]
                    if val != 0:
                        u_row = (i_start + li) * m + (j_start + lj)
                        U[u_row][col] = val
                        
            # Update V
            for lj in range(2):
                for lk in range(2):
                    val = V_strassen[t][lj * 2 + lk]
                    if val != 0:
                        v_row = (j_start + lj) * p + (k_start + lk)
                        V[v_row][col] = val
                        
            # Update W
            for lk in range(2):
                for li in range(2):
                    val = W_strassen[lk * 2 + li][t]
                    if val != 0:
                        w_row = (k_start + lk) * n + (i_start + li)
                        W[w_row][col] = val
                        
            col += 1

    def add_schoolbook_block(i_start, j_start, k_start, i_end, j_end, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u_row = i * m + j
                    v_row = j * p + k
                    w_row = k * n + i
                    U[u_row][col] = 1
                    V[v_row][col] = 1
                    W[w_row][col] = 1
                    col += 1

    # Process i in {0, 1} with 8 Strassen blocks
    # j pairs: [0,1], [2,3]
    # k pairs: [0,1], [2,3], [4,5], [6,7]
    
    j_pairs = [(0, 2), (2, 4)]
    k_pairs = [(0, 2), (2, 4), (4, 6), (6, 8)]
    
    for i_start in [0]: # Only i=0,1 block, so start 0, size 2
        for j_start, j_end in j_pairs:
            for k_start, k_end in k_pairs:
                add_strassen_block(i_start, j_start, k_start)
                
    # Process i = 2 with schoolbook
    add_schoolbook_block(2, 0, 0, 3, 4, 8)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
