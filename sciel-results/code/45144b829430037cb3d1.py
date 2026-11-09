"""Matrix multiplication 4x4x5 rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

# Deadline guard: must return within 120s, use 110s as safety margin
_START_TIME = time.time()
_DEADLINE = 110.0

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Strategy: split-column decomposition
    # 4x4x5 tensor = 4x4x4 tensor (columns 0-3) + 4x4x1 tensor (column 4)
    # 4x4x4: Use recursive Strassen (rank 49)
    # 4x4x1: Use schoolbook (rank 16)
    # Total rank = 49 + 16 = 65
    
    # Verify time budget
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback to schoolbook if somehow we're over (shouldn't happen for this deterministic code)
        return _schoolbook_decomposition(n, m, p)
    
    total_rank = 65
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * total_rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 65
    
    # --- Part 1: 4x4x4 Strassen decomposition (rank 49) ---
    # Uses recursive Strassen: 4x4 = 4 block multiplications of 2x2
    # Each 2x2 multiplication uses rank-7 Strassen
    # Total: 7 * 7 = 49 rank-1 terms
    
    # Strassen 2x2 decomposition vectors
    # For multiplying 2x2 matrices A, B to get C:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U_2x2[r][i]: coefficient of A[i] in linear form r (i=0,1,2,3 for A00,A01,A10,A11)
    U_2x2 = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [1, 0, -1, 0],  # M6: A10-A00
        [0, 1, 0, -1],  # M7: A01-A11
    ]
    
    # V_2x2[r][j]: coefficient of B[j] in linear form r (j=0,1,2,3 for B00,B01,B10,B11)
    V_2x2 = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [0, 0, 1, -1],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1],   # M7: B10+B11
    ]
    
    # W_2x2[r][c]: coefficient in C[c] (c=0,1,2,3 for C00,C01,C10,C11)
    W_2x2 = [
        [1, 0, 0, 1],   # M1: C00+C11
        [0, 0, 1, -1],  # M2: C10-C11
        [0, 1, 0, 1],   # M3: C01+C11
        [1, 0, 1, 0],   # M4: C00+C10
        [-1, 1, 0, 0],  # M5: -C00+C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0],   # M7: C00
    ]
    
    col = 0
    # For the 4x4x4 part, we use recursive Strassen
    # 4x4 matrix A is partitioned into 4 blocks: A00, A01, A10, A11 (each 2x2)
    # Similarly for B and C
    
    # The 7 Strassen linear forms on the 4x4 A (treating it as 2x2 blocks of 2x2):
    # S_A_m is a 2x2 matrix of 2x2 blocks
    # Each entry S_A_m[i][j] is a linear combination of the 2x2 blocks of A
    
    # For the outer Strassen (4x4 level), the linear forms on A are:
    # SA_m[i1][j1] = sum over blocks with coefficients
    # Actually, let's think of it differently.
    
    # 4x4 A = [A00 A01; A10 A11] where each Aij is 2x2
    # We want to compute C = A * B for 4x4 matrices
    # C = [C00 C01; C10 C11] where each Cij is 2x2
    
    # Strassen at the 4x4 level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each M_k is a 2x2 matrix product
    # We decompose each 2x2 product using the rank-7 Strassen
    
    # So for each outer term m (0-6), and each inner term r (0-6):
    # The linear form on A: u_m ⊗ u_r (tensor product of the 2x2 block form and the 2x2 inner form)
    # The linear form on B: v_m ⊗ v_r
    # The output: w_r * (coeff of M_m in C)
    
    # Let's define the outer Strassen coefficients
    # SA_m: 2x2 matrix of coefficients for the 4 blocks of A
    # SA_m[i][j] = coefficient of block Aij in S_A_m
    
    SA = [
        [[1, 0], [0, 1]],   # M1: A00 + A11
        [[0, 0], [1, 1]],   # M2: A10 + A11
        [[1, 0], [0, 0]],   # M3: A00
        [[0, 0], [0, 1]],   # M4: A11
        [[1, 1], [0, 0]],   # M5: A00 + A01
        [[1, 0], [-1, 0]],  # M6: A10 - A00
        [[0, 1], [0, -1]],  # M7: A01 - A11
    ]
    
    SB = [
        [[1, 0], [0, 1]],   # M1: B00 + B11
        [[1, 0], [0, 0]],   # M2: B00
        [[0, 1], [0, -1]],  # M3: B01 - B11
        [[0, 0], [1, -1]],  # M4: B10 - B00
        [[0, 0], [0, 1]],   # M5: B11
        [[1, 1], [0, 0]],   # M6: B00 + B01
        [[0, 0], [1, 1]],   # M7: B10 + B11
    ]
    
    # SC_m: 2x2 matrix of coefficients for the 4 blocks of C
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    SC = [
        [[1, 0], [0, 1]],   # M1: C00 + C11
        [[0, 0], [1, -1]],  # M2: C10 - C11
        [[0, 1], [0, 1]],   # M3: C01 + C11
        [[1, 0], [1, 0]],   # M4: C00 + C10
        [[-1, 1], [0, 0]],  # M5: -C00 + C01
        [[0, 0], [0, 1]],   # M6: C11
        [[1, 0], [0, 0]],   # M7: C00
    ]
    
    # Now for each outer term m and inner term r, we have a rank-1 decomposition term
    # The linear form on A: for block (bi, bj) of A, and position (pi, pj) within the block
    # The coefficient is SA[m][bi][bj] * U_2x2[r][pi*2+pj]
    # Wait, I need to be more careful with indexing.
    
    # A is 4x4. Index i = 0..3, j = 0..3
    # Block index: bi = i // 2, bj = j // 2
    # Position within block: pi = i % 2, pj = j % 2
    
    # The 2x2 inner matrix for block (bi, bj) has entries A[2*bi+pi, 2*bj+pj]
    # The linear form U_2x2[r] applies to this 2x2 block
    # U_2x2[r][k] where k = 0,1,2,3 corresponds to positions (0,0), (0,1), (1,0), (1,1)
    # So k = pi*2 + pj
    
    for m_idx in range(7):
        for r_idx in range(7):
            # Build U vector (length 16)
            for i in range(4):
                for j in range(4):
                    bi, bj = i // 2, j // 2
                    pi, pj = i % 2, j % 2
                    k = pi * 2 + pj
                    coeff = SA[m_idx][bi][bj] * U_2x2[r_idx][k]
                    U[i * m + j][col] = coeff
            
            # Build V vector (length 20, but only first 16 entries for 4x4 part, last 4 for column 4)
            # For the 4x4x4 part, B is 4x4 (m=4, p=4 for this sub-problem)
            # B index: j=0..3, k=0..3
            # Block index: bi = j // 2, bj = k // 2
            # Position within block: pi = j % 2, pj = k % 2
            for j in range(4):
                for k in range(4):
                    bi, bj = j // 2, k // 2
                    pi, pj = j % 2, k % 2
                    inner_k = pi * 2 + pj
                    coeff = SB[m_idx][bi][bj] * V_2x2[r_idx][inner_k]
                    V[j * p + k][col] = coeff
            
            # Build W vector (length 20)
            # C index: k=0..3 (for 4x4 part), i=0..3
            # W[k*n+i] where n=4
            # Block index: bi = i // 2, bj = k // 2
            # Position within block: pi = i % 2, pj = k % 2
            for k in range(4):
                for i in range(4):
                    bi, bj = i // 2, k // 2
                    pi, pj = i % 2, k % 2
                    inner_k = pi * 2 + pj
                    coeff = SC[m_idx][bi][bj] * W_2x2[r_idx][inner_k]
                    W[k * n + i][col] = coeff
            
            col += 1
    
    # --- Part 2: 4x4x1 schoolbook decomposition (rank 16) ---
    # For the 5th column (index 4), we use schoolbook
    # C[i][4] = sum_{j=0}^{3} A[i][j] * B[j][4]
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1  # B[j][4]
            W[4 * n + i][col] = 1  # C[4][i]
            col += 1
    
    return {"U": U, "V": V, "W": W}


def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
