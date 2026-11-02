"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time

def construct():
    # Strassen-based decomposition for 2x2 * 2x16
    # n=2, m=2, p=16
    # Use 8 blocks of 2 columns each. For each block, use 7 Strassen terms.
    # Total rank = 8 * 7 = 56
    
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks of 2 columns
    rank = num_blocks * 7  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    for block in range(num_blocks):
        # block covers columns 2*block and 2*block+1
        c0 = 2 * block
        c1 = 2 * block + 1
        base_col = block * 7
        
        # Strassen terms for 2x2 matrix multiplication:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B01 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M6
        # C11 = M1 - M2 + M3 + M6
        
        # Term 1: M1 = (A00 + A11) * (B00 + B11)
        col = base_col + 0
        U[0][col] = 1  # A00
        U[3][col] = 1  # A11
        V[0 * p + c0][col] = 1  # B0,c0
        V[1 * p + c0][col] = 1  # B1,c0
        V[0 * p + c1][col] = 1  # B0,c1
        V[1 * p + c1][col] = 1  # B1,c1
        W[c0 * n + 0][col] = 1  # C0,c0
        W[c1 * n + 1][col] = 1  # C1,c1
        
        # Term 2: M2 = (A10 + A11) * B00
        col = base_col + 1
        U[2][col] = 1  # A10
        U[3][col] = 1  # A11
        V[0 * p + c0][col] = 1  # B0,c0
        W[c0 * n + 1][col] = 1  # C1,c0
        
        # Term 3: M3 = A00 * (B10 - B11)
        col = base_col + 2
        U[0][col] = 1  # A00
        V[1 * p + c0][col] = 1  # B1,c0
        V[1 * p + c1][col] = -1  # -B1,c1
        W[c1 * n + 0][col] = 1  # C0,c1
        W[c1 * n + 1][col] = 1  # C1,c1
        
        # Term 4: M4 = A11 * (B01 - B00)
        col = base_col + 3
        U[3][col] = 1  # A11
        V[0 * p + c1][col] = 1  # B0,c1
        V[0 * p + c0][col] = -1  # -B0,c0
        W[c0 * n + 0][col] = 1  # C0,c0
        W[c0 * n + 1][col] = 1  # C1,c0
        
        # Term 5: M5 = (A00 + A01) * B11
        col = base_col + 4
        U[0][col] = 1  # A00
        U[1][col] = 1  # A01
        V[1 * p + c1][col] = 1  # B1,c1
        W[c0 * n + 0][col] = -1  # -C0,c0
        W[c1 * n + 0][col] = 1  # C0,c1
        
        # Term 6: M6 = (A10 - A00) * (B00 + B01)
        col = base_col + 5
        U[2][col] = 1  # A10
        U[0][col] = -1  # -A00
        V[0 * p + c0][col] = 1  # B0,c0
        V[0 * p + c1][col] = 1  # B0,c1
        W[c1 * n + 1][col] = 1  # C1,c1
        
        # Term 7: M7 = (A01 - A11) * (B10 + B11)
        col = base_col + 6
        U[1][col] = 1  # A01
        U[3][col] = -1  # -A11
        V[1 * p + c0][col] = 1  # B1,c0
        V[1 * p + c1][col] = 1  # B1,c1
        W[c0 * n + 0][col] = 1  # C0,c0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
