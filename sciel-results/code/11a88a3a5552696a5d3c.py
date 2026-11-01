"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 25 decomposition using Strassen for 2x2 blocks and schoolbook for 2x1 block
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a 2x2x2 Strassen block at columns k0, k1 with offset col
    def add_strassen_block(k0, k1, col_offset):
        # M1: (A00+A11)(B00+B11) -> C00+C11
        # U: A00(0), A11(3)
        U[0][col_offset] = 1
        U[3][col_offset] = 1
        # V: B00(0*p+k0), B11(1*p+k1)
        V[0 * p + k0][col_offset] = 1
        V[1 * p + k1][col_offset] = 1
        # W: C00(k0*n+0), C11(k1*n+1)
        W[k0 * n + 0][col_offset] = 1
        W[k1 * n + 1][col_offset] = 1
        
        # M2: (A10+A11)B00 -> C10
        U[2][col_offset + 1] = 1
        U[3][col_offset + 1] = 1
        V[0 * p + k0][col_offset + 1] = 1
        W[k0 * n + 1][col_offset + 1] = 1
        
        # M3: A00(B01-B11) -> C01
        U[0][col_offset + 2] = 1
        V[0 * p + k1][col_offset + 2] = 1
        V[1 * p + k1][col_offset + 2] = -1
        W[k1 * n + 0][col_offset + 2] = 1
        
        # M4: A11(B10-B00) -> C10
        U[3][col_offset + 3] = 1
        V[1 * p + k0][col_offset + 3] = 1
        V[0 * p + k0][col_offset + 3] = -1
        W[k0 * n + 1][col_offset + 3] = 1
        
        # M5: (A00+A01)B11 -> C00(-1)+C01(1)
        U[0][col_offset + 4] = 1
        U[1][col_offset + 4] = 1
        V[1 * p + k1][col_offset + 4] = 1
        W[k0 * n + 0][col_offset + 4] = -1
        W[k1 * n + 0][col_offset + 4] = 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        U[2][col_offset + 5] = 1
        U[0][col_offset + 5] = -1
        V[0 * p + k0][col_offset + 5] = 1
        V[0 * p + k1][col_offset + 5] = 1
        W[k1 * n + 1][col_offset + 5] = 1
        
        # M7: (A01-A11)(B10+B11) -> C01
        U[1][col_offset + 6] = 1
        U[3][col_offset + 6] = -1
        V[1 * p + k0][col_offset + 6] = 1
        V[1 * p + k1][col_offset + 6] = 1
        W[k1 * n + 0][col_offset + 6] = 1

    # Helper to add a 2x2x1 schoolbook block at column k0 with offset col
    def add_schoolbook_block(k0, col_offset):
        # M1: A00 B00 -> C00
        U[0][col_offset] = 1
        V[0 * p + k0][col_offset] = 1
        W[k0 * n + 0][col_offset] = 1
        
        # M2: A01 B10 -> C00
        U[1][col_offset + 1] = 1
        V[1 * p + k0][col_offset + 1] = 1
        W[k0 * n + 0][col_offset + 1] = 1
        
        # M3: A10 B00 -> C10
        U[2][col_offset + 2] = 1
        V[0 * p + k0][col_offset + 2] = 1
        W[k0 * n + 1][col_offset + 2] = 1
        
        # M4: A11 B10 -> C10
        U[3][col_offset + 3] = 1
        V[1 * p + k0][col_offset + 3] = 1
        W[k0 * n + 1][col_offset + 3] = 1

    # Block 1: k=0,1
    add_strassen_block(0, 1, 0)
    # Block 2: k=2,3
    add_strassen_block(2, 3, 7)
    # Block 3: k=4,5
    add_strassen_block(4, 5, 14)
    # Block 4: k=6
    add_schoolbook_block(6, 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
