import time
import numpy as np

DIMENSIONS = [4, 4, 5]
TIME_LIMIT = 150
SAFETY_MARGIN = 10

# EVOLVE-BLOCK-START
def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Construct Strassen 4x4x4 decomposition (rank 49)
    try:
        if time.time() - start_time < TIME_LIMIT - SAFETY_MARGIN:
            result_444 = _strassen_4x4x4()
            # Extend to 4x4x5 by adding schoolbook terms for the 5th column
            result = _extend_to_445(result_444, n, m, p)
            
            if _verify_decomposition(result, n, m, p):
                return result
    except Exception as e:
        # Fallback to schoolbook if construction fails
        pass
    
    # Fallback: schoolbook decomposition (rank 80)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Return U, V, W (each 4x7) for Strassen 2x2x2 matrix multiplication.
    
    U[i*2+j] corresponds to A[i,j]
    V[j*2+k] corresponds to B[j,k]
    W[k*2+i] corresponds to C[i,k]
    
    Strassen formulas:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B10 - B00)
    M4 = A11 * (B01 - B11)
    M5 = (A00 + A01) * B11
    M6 = (A01 - A11) * (B00 + B01)
    M7 = (A00 - A10) * (B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00 += M1
    W[3][0] = 1  # C11 += M1
    
    # M2 = (A10 + A11) * B00
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10 += M2
    
    # M3 = A00 * (B10 - B00)
    U[0][2] = 1  # A00
    V[2][2] = 1  # B10
    V[0][2] = -1 # B00
    W[1][2] = 1  # C01 += M3
    W[3][2] = 1  # C11 += M3
    
    # M4 = A11 * (B01 - B11)
    U[3][3] = 1  # A11
    V[1][3] = 1  # B01
    V[3][3] = -1 # B11
    W[0][3] = 1  # C00 += M4
    W[2][3] = 1  # C10 += M4
    
    # M5 = (A00 + A01) * B11
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    W[1][4] = 1  # C01 += M5
    W[0][4] = -1 # C00 -= M5
    
    # M6 = (A01 - A11) * (B00 + B01)
    U[1][5] = 1  # A01
    U[3][5] = -1 # A11
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    W[3][5] = 1  # C11 += M6
    
    # M7 = (A00 - A10) * (B10 + B11)
    U[0][6] = 1  # A00
    U[2][6] = -1 # A10
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    W[0][6] = 1  # C00 += M7
    
    return U, V, W

def _strassen_4x4x4():
    """Construct rank 49 decomposition for 4x4x4 using block Strassen."""
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # Strassen 2x2x2: verify it's correct first
    # M1 = (A00+A11)*(B00+B11) -> C00+=M1, C11+=M1
    # M2 = (A10+A11)*B00 -> C10+=M2
    # M3 = A00*(B10-B00) -> C01+=M3, C11+=M3
    # M4 = A11*(B01-B11) -> C00+=M4, C10+=M4
    # M5 = (A00+A01)*B11 -> C01+=M5, C00-=M5
    # M6 = (A01-A11)*(B00+B01) -> C11+=M6
    # M7 = (A00-A10)*(B10+B11) -> C00+=M7
    
    # Verify C formulas:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Block Strassen for 4x4x4:
    # Split A, B, C into 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = [[C00, C01], [C10, C11]]
    
    # 7 block-level Strassen steps:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B10-B00)
    # M4 = A11 * (B01-B11)
    # M5 = (A00+A01) * B11
    # M6 = (A01-A11) * (B00+B01)
    # M7 = (A00-A10) * (B10+B11)
    
    # C block contributions:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each block product Mi = Ai * Bi where Ai, Bi are 2x2 matrices
    # Each 2x2x2 multiplication uses rank 7 (Strassen)
    # So total rank = 7 * 7 = 49
    
    # For each block-level step r (0-6) and each inner Strassen step s (0-6):
    # col = r*7 + s
    
    # A coefficients for each step: dict from (block_row, block_col) to coeff
    # B coefficients: dict from (block_col, block_row) to coeff  (B is [j,k] so block is (bj, bk))
    # C coefficients: dict from (block_row, block_col) to coeff  (C is [i,k] so block is (ci, ck))
    
    a_steps = [
        {(0,0): 1, (1,1): 1},       # M1: A00 + A11
        {(1,0): 1, (1,1): 1},       # M2: A10 + A11
        {(0,0): 1},                  # M3: A00
        {(1,1): 1},                  # M4: A11
        {(0,0): 1, (0,1): 1},       # M5: A00 + A01
        {(0,1): 1, (1,1): -1},      # M6: A01 - A11
        {(0,0): 1, (1,0): -1},      # M7: A00 - A10
    ]
    
    b_steps = [
        {(0,0): 1, (1,1): 1},       # M1: B00 + B11
        {(0,0): 1},                  # M2: B00
        {(1,0): 1, (0,0): -1},      # M3: B10 - B00
        {(0,1): 1, (1,1): -1},      # M4: B01 - B11
        {(1,1): 1},                  # M5: B11
        {(0,0): 1, (0,1): 1},       # M6: B00 + B01
        {(1,0): 1, (1,1): 1},       # M7: B10 + B11
    ]
    
    c_steps = [
        {(0,0): 1, (1,1): 1},       # M1: C00 + C11
        {(1,0): 1},                  # M2: C10
        {(0,1): 1, (1,1): 1},       # M3: C01 + C11
        {(0,0): 1, (1,0): 1},       # M4: C00 + C10
        {(0,1): 1, (0,0): -1},      # M5: C01 - C00
        {(1,1): 1},                  # M6: C11
        {(0,0): 1},                  # M7: C00
    ]
    
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            
            # Process U (A matrix): U[i*4+j]
            for (bi, bj), coeff_a in a_steps[r].items():
                if coeff_a == 0:
                    continue
                for li in range(2):
                    for lj in range(2):
                        local_u = U2[li*2 + lj][s]
                        if local_u == 0:
                            continue
                        global_i = bi * 2 + li
                        global_j = bj * 2 + lj
                        U[global_i * 4 + global_j][col] += coeff_a * local_u
            
            # Process V (B matrix): V[j*4+k]
            for (bj, bk), coeff_b in b_steps[r].items():
                if coeff_b == 0:
                    continue
                for lj in range(2):
                    for lk in range(2):
                        local_v = V2[lj*2 + lk][s]
                        if local_v == 0:
                            continue
                        global_j = bj * 2 + lj
                        global_k = bk * 2 + lk
                        V[global_j * 4 + global_k][col] += coeff_b * local_v
            
            # Process W (C matrix): W[k*4+i]
            for (ci, ck), coeff_c in c_steps[r].items():
                if coeff_c == 0:
                    continue
                for li in range(2):
                    for lk in range(2):
                        # W2 index: k*2+i, so W2[lk*2+li]
                        local_w = W2[lk*2 + li][s]
                        if local_w == 0:
                            continue
                        global_k = ck * 2 + lk
                        global_i = ci * 2 + li
                        W[global_k * 4 + global_i][col] += coeff_c * local_w
    
    return {"U": U, "V": V, "W": W}

def _extend_to_445(decomp_444, n, m, p):
    """Extend 4x4x4 decomposition to 4x4x5."""
    U4 = decomp_444["U"]
    V4 = decomp_444["V"]
    W4 = decomp_444["W"]
    rank_444 = len(U4[0])
    
    # New dimensions
    new_rank = rank_444 + 4
    U = [[0] * new_rank for _ in range(n * m)]
    V = [[0] * new_rank for _ in range(m * p)]
    W = [[0] * new_rank for _ in range(p * n)]
    
    # Copy 4x4x4 part
    # U: 16 x 49 -> 16 x 53
    for i in range(16):
        for r in range(rank_444):
            U[i][r] = U4[i][r]
            
    # V: 16 x 49 -> 20 x 53
    # V4 is for B[0:4, 0:4].
    # V is for B[0:4, 0:5].
    # Columns 0-3 of B are handled. Column 4 is schoolbook.
    for j in range(4):
        for k in range(4):
            for r in range(rank_444):
                V[j*5 + k][r] = V4[j*4 + k][r]
                
    # W: 16 x 49 -> 20 x 53
    # W4 is for C[0:4, 0:4].
    # W is for C[0:5, 0:4].
    for k in range(4):
        for i in range(4):
            for r in range(rank_444):
                W[k*4 + i][r] = W4[k*4 + i][r]
                
    # Add schoolbook for 5th column (k=4)
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    for j in range(4):
        col = rank_444 + j
        for i in range(4):
            U[i*4 + j][col] = 1
        V[j*5 + 4][col] = 1
        for i in range(4):
            W[4*4 + i][col] = 1
            
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(decomp, n, m, p, num_checks=200):
    """Verify the decomposition using numpy for speed."""
    import random
    U = np.array(decomp["U"], dtype=np.float64)
    V = np.array(decomp["V"], dtype=np.float64)
    W = np.array(decomp["W"], dtype=np.float64)
    rank = U.shape[1]
    
    # Check all valid (i,j,k) triples give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1) > 1e-9:
                    return False
    
    # Check some invalid positions give 0
    random.seed(42)
    for _ in range(num_checks):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        i = a // m; j = a % m
        j2 = b // p; k = b % p
        k2 = c // n; i2 = c % n
        if i == i2 and j == j2 and k == k2:
            continue
        val = np.dot(U[a], V[b] * W[c])
        if abs(val) > 1e-9:
            return False
    
    return True
# EVOLVE-BLOCK-END
