"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(A_rows, A_cols, B_rows, B_cols, W_rows, W_cols):
        nonlocal col
        # A_rows: [r0, r1], A_cols: [c0, c1]
        # B_rows: [r0, r1], B_cols: [c0, c1]
        # W_rows: [r0, r1], W_cols: [c0, c1]
        
        # Indices
        A00_idx = A_rows[0] * m + A_cols[0]
        A01_idx = A_rows[0] * m + A_cols[1]
        A10_idx = A_rows[1] * m + A_cols[0]
        A11_idx = A_rows[1] * m + A_cols[1]
        
        B00_idx = B_rows[0] * p + B_cols[0]
        B01_idx = B_rows[0] * p + B_cols[1]
        B10_idx = B_rows[1] * p + B_cols[0]
        B11_idx = B_rows[1] * p + B_cols[1]
        
        W00_idx = W_cols[0] * n + W_rows[0]
        W01_idx = W_cols[1] * n + W_rows[0]
        W10_idx = W_cols[0] * n + W_rows[1]
        W11_idx = W_cols[1] * n + W_rows[1]
        
        # M1: (A00+A11)(B00+B11) -> W00+1, W11+1
        U[A00_idx][col] += 1
        U[A11_idx][col] += 1
        V[B00_idx][col] += 1
        V[B11_idx][col] += 1
        W[W00_idx][col] += 1
        W[W11_idx][col] += 1
        col += 1
        
        # M2: (A10+A11)B00 -> W10+1, W11-1
        U[A10_idx][col] += 1
        U[A11_idx][col] += 1
        V[B00_idx][col] += 1
        W[W10_idx][col] += 1
        W[W11_idx][col] -= 1
        col += 1
        
        # M3: A00(B10-B11) -> W01+1, W11+1
        U[A00_idx][col] += 1
        V[B10_idx][col] += 1
        V[B11_idx][col] -= 1
        W[W01_idx][col] += 1
        W[W11_idx][col] += 1
        col += 1
        
        # M4: A11(B01-B00) -> W00+1, W10+1
        U[A11_idx][col] += 1
        V[B01_idx][col] += 1
        V[B00_idx][col] -= 1
        W[W00_idx][col] += 1
        W[W10_idx][col] += 1
        col += 1
        
        # M5: (A00+A10)B11 -> W00-1, W01+1
        U[A00_idx][col] += 1
        U[A10_idx][col] += 1
        V[B11_idx][col] += 1
        W[W00_idx][col] -= 1
        W[W01_idx][col] += 1
        col += 1
        
        # M6: (A11-A00)(B00+B01) -> W11+1
        U[A11_idx][col] += 1
        U[A00_idx][col] -= 1
        V[B00_idx][col] += 1
        V[B01_idx][col] += 1
        W[W11_idx][col] += 1
        col += 1
        
        # M7: (A01-A11)(B10+B11) -> W00+1, W11+1
        U[A01_idx][col] += 1
        U[A11_idx][col] -= 1
        V[B10_idx][col] += 1
        V[B11_idx][col] += 1
        W[W00_idx][col] += 1
        W[W11_idx][col] += 1
        col += 1

    def add_schoolbook_221(A_rows, A_cols, B_rows, B_col, W_rows, W_col):
        nonlocal col
        A00_idx = A_rows[0] * m + A_cols[0]
        A01_idx = A_rows[0] * m + A_cols[1]
        A10_idx = A_rows[1] * m + A_cols[0]
        A11_idx = A_rows[1] * m + A_cols[1]
        
        B0_idx = B_rows[0] * p + B_col
        B1_idx = B_rows[1] * p + B_col
        
        W0_idx = W_col * n + W_rows[0]
        W1_idx = W_col * n + W_rows[1]
        
        # A00 B0 -> W0
        U[A00_idx][col] += 1
        V[B0_idx][col] += 1
        W[W0_idx][col] += 1
        col += 1
        
        # A01 B1 -> W0
        U[A01_idx][col] += 1
        V[B1_idx][col] += 1
        W[W0_idx][col] += 1
        col += 1
        
        # A10 B0 -> W1
        U[A10_idx][col] += 1
        V[B0_idx][col] += 1
        W[W1_idx][col] += 1
        col += 1
        
        # A11 B1 -> W1
        U[A11_idx][col] += 1
        V[B1_idx][col] += 1
        W[W1_idx][col] += 1
        col += 1

    # Sub-problems
    # A_L: Rows 0-1, Cols 0-1
    # A_R: Rows 0-1, Cols 2-3
    
    # B_1T: Rows 0-1, Cols 0-3 (Split into 0-1, 2-3)
    # B_1B: Rows 2-3, Cols 0-3 (Split into 0-1, 2-3)
    
    # B_2aT: Rows 0-1, Cols 4-5
    # B_2aB: Rows 2-3, Cols 4-5
    
    # B_2bT: Rows 0-1, Col 6
    # B_2bB: Rows 2-3, Col 6
    
    # 1. A_L B_1TL (2x2x2)
    add_strassen_222([0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # 2. A_L B_1TR (2x2x2)
    add_strassen_222([0,1], [0,1], [0,1], [2,3], [0,1], [2,3])
    
    # 3. A_R B_1BL (2x2x2)
    add_strassen_222([0,1], [2,3], [2,3], [0,1], [0,1], [0,1])
    
    # 4. A_R B_1BR (2x2x2)
    add_strassen_222([0,1], [2,3], [2,3], [2,3], [0,1], [2,3])
    
    # 5. A_L B_2aT (2x2x2)
    add_strassen_222([0,1], [0,1], [0,1], [4,5], [0,1], [4,5])
    
    # 6. A_R B_2aB (2x2x2)
    add_strassen_222([0,1], [2,3], [2,3], [4,5], [0,1], [4,5])
    
    # 7. A_L B_2bT (2x2x1)
    add_schoolbook_221([0,1], [0,1], [0,1], 6, [0,1], 6)
    
    # 8. A_R B_2bB (2x2x1)
    add_schoolbook_221([0,1], [2,3], [2,3], 6, [0,1], 6)
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
