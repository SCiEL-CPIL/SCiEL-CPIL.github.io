"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    
    Strategy: 
    1. Split p=8 into two blocks of p=4.
    2. Each block computes C_block = A * B_block (2x3 * 3x4 -> 2x4).
    3. Decompose 2x3x4 by splitting m=3 into m1=1 and m2=2.
       Part 1: 2x1x4 (Rank 8).
       Part 2: 2x2x4. Split p=4 into two blocks of p=2.
               Each 2x2x2 uses Strassen (Rank 7).
               Total Rank for 2x2x4: 14.
       Total Rank for 2x3x4: 8 + 14 = 22.
    4. Total Rank: 2 * 22 = 44.
    
    This construction follows the directive to use p-splitting.
    """
    start_time = time.time()
    timeout = 140.0  # Safety margin

    n, m, p = DIMENSIONS  # 2, 3, 8
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44

    col = 0

    # Helper to add a 2x2x2 Strassen block
    # A indices: a00, a01, a10, a11 (global u_idx)
    # B indices: b00, b01, b10, b11 (global v_idx)
    # C indices: c00, c01, c10, c11 (global w_idx)
    def add_strassen_2x2x2(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal col
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1

        # M2 = (A10 + A11) * B00
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1

        # M3 = A00 * (B01 - B11)
        U[a00][col] += 1
        V[b01][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1

        # M4 = A11 * (B10 - B00)
        U[a11][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1

        # M5 = (A00 + A01) * B11
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1

        # M6 = (A10 - A00) * (B00 + B01)
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] += 1
        col += 1

        # M7 = (A01 - A11) * (B10 + B11)
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1

    # Split p=8 into two blocks of 4: [0:4] and [4:8]
    for p_block in range(2):
        p_start = p_block * 4
        
        # --- Part 1 of this block: 2x1x4 (j=0) ---
        # A[:, 0] * B[0, p_start:p_start+4]
        # Rank 8.
        for i in range(n):
            for k in range(4):
                u_idx = i * m + 0
                v_idx = 0 * p + (p_start + k)
                w_idx = (p_start + k) * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
        
        # --- Part 2 of this block: 2x2x4 (j=1,2) ---
        # A[:, 1:3] * B[1:3, p_start:p_start+4]
        # Split p=4 into two blocks of 2: [p_start, p_start+2) and [p_start+2, p_start+4)
        
        # A indices (global):
        # A00: A[0, 1] -> u_idx = 1
        # A01: A[0, 2] -> u_idx = 2
        # A10: A[1, 1] -> u_idx = 4
        # A11: A[1, 2] -> u_idx = 5
        
        a00 = 1
        a01 = 2
        a10 = 4
        a11 = 5

        # Sub-block 1: p indices p_start, p_start+1
        # B indices (global):
        # B00: B[1, p_start] -> v_idx = 1*p + p_start
        # B01: B[1, p_start+1] -> v_idx = 1*p + p_start + 1
        # B10: B[2, p_start] -> v_idx = 2*p + p_start
        # B11: B[2, p_start+1] -> v_idx = 2*p + p_start + 1
        
        b00_1 = 1*p + p_start
        b01_1 = 1*p + p_start + 1
        b10_1 = 2*p + p_start
        b11_1 = 2*p + p_start + 1
        
        # C indices (global):
        # C00: C[0, p_start] -> w_idx = p_start*n + 0
        # C01: C[0, p_start+1] -> w_idx = (p_start+1)*n + 0
        # C10: C[1, p_start] -> w_idx = p_start*n + 1
        # C11: C[1, p_start+1] -> w_idx = (p_start+1)*n + 1
        
        c00_1 = p_start*n + 0
        c01_1 = (p_start+1)*n + 0
        c10_1 = p_start*n + 1
        c11_1 = (p_start+1)*n + 1

        add_strassen_2x2x2(a00, a01, a10, a11, b00_1, b01_1, b10_1, b11_1, c00_1, c01_1, c10_1, c11_1)

        # Sub-block 2: p indices p_start+2, p_start+3
        # B indices (global):
        # B00: B[1, p_start+2] -> v_idx = 1*p + p_start + 2
        # B01: B[1, p_start+3] -> v_idx = 1*p + p_start + 3
        # B10: B[2, p_start+2] -> v_idx = 2*p + p_start + 2
        # B11: B[2, p_start+3] -> v_idx = 2*p + p_start + 3
        
        b00_2 = 1*p + p_start + 2
        b01_2 = 1*p + p_start + 3
        b10_2 = 2*p + p_start + 2
        b11_2 = 2*p + p_start + 3
        
        # C indices (global):
        # C00: C[0, p_start+2] -> w_idx = (p_start+2)*n + 0
        # C01: C[0, p_start+3] -> w_idx = (p_start+3)*n + 0
        # C10: C[1, p_start+2] -> w_idx = (p_start+2)*n + 1
        # C11: C[1, p_start+3] -> w_idx = (p_start+3)*n + 1
        
        c00_2 = (p_start+2)*n + 0
        c01_2 = (p_start+3)*n + 0
        c10_2 = (p_start+2)*n + 1
        c11_2 = (p_start+3)*n + 1

        add_strassen_2x2x2(a00, a01, a10, a11, b00_2, b01_2, b10_2, b11_2, c00_2, c01_2, c10_2, c11_2)

    # Check time
    if time.time() - start_time > timeout:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
