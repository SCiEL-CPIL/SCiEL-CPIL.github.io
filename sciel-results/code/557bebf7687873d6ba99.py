import time

_start_time = time.time()
_DEADLINE = 120.0

def _time_remaining():
    return _DEADLINE - (time.time() - _start_time)

def construct():
    """
    Returns a rank-42 exact tensor decomposition for 2x2 * 2x12 matrix multiplication.
    
    The 12 columns of the second matrix are divided into 6 blocks of 2 columns each.
    Each block multiplication is a standard 2x2 * 2x2 multiplication, which has rank 7 (Strassen).
    Since the blocks are independent, the total rank is 6 * 7 = 42.
    """
    if _time_remaining() < 1.0:
        return None

    n, m, p = 2, 2, 12
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42

    # U: (n*m, rank) = (4, 42)
    # V: (m*p, rank) = (24, 42)
    # W: (p*n, rank) = (24, 42)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 7 products for 2x2 * 2x2:
    # Indices for U (A): (i, j) -> row i*m + j
    # Indices for V (B): (j, k) -> row j*p + k (local k is 0 or 1)
    # Indices for W (C): (k, i) -> row k*n + i (local k is 0 or 1)
    
    # Strassen coefficients:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # Reconstruction:
    # C00 = M1 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6

    # U_coeffs[s] = list of (i, j, coeff)
    U_coeffs = [
        [(0, 0, 1), (1, 1, 1)],           # M1
        [(1, 0, 1), (1, 1, 1)],           # M2
        [(0, 0, 1)],                      # M3
        [(1, 1, 1)],                      # M4
        [(0, 0, 1), (0, 1, 1)],           # M5
        [(1, 0, 1), (0, 0, -1)],          # M6
        [(0, 1, 1), (1, 1, -1)],          # M7
    ]
    
    # V_coeffs[s] = list of (j, k, coeff)
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],           # M1
        [(0, 0, 1)],                      # M2
        [(0, 1, 1), (1, 1, -1)],          # M3
        [(1, 0, 1), (0, 0, -1)],          # M4
        [(1, 1, 1)],                      # M5
        [(0, 0, 1), (0, 1, 1)],           # M6
        [(1, 0, 1), (1, 1, 1)],           # M7
    ]
    
    # W_coeffs[s] = list of (k, i, coeff)
    # Note: W[k*n+i, r]
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],           # M1 -> C00 += 1, C11 += 1
        [(0, 0, -1), (1, 1, 1)],          # M2 -> C10 += 1, C11 += -1 ? 
        # Wait, let's re-verify M2 contribution.
        # M2 = (A10+A11)*B00
        # C10 = M2 + M4
        # C11 = M1 - M2 + M6
        # So M2 contributes +1 to C10 and -1 to C11.
        # C10 is k=0, i=1. C11 is k=1, i=1.
        # So W coeffs for M2: (0, 1, 1), (1, 1, -1).
        
        # Let's re-verify all W coeffs carefully.
        # C00 = M1 - M5 + M7
        #   M1: (0,0, 1)
        #   M5: (0,0, -1)
        #   M7: (0,0, 1)
        # C01 = M3 + M5
        #   M3: (1,0, 1)  (k=1, i=0)
        #   M5: (1,0, 1)
        # C10 = M2 + M4
        #   M2: (0,1, 1)  (k=0, i=1)
        #   M4: (0,1, 1)
        # C11 = M1 - M2 + M6
        #   M1: (1,1, 1)  (k=1, i=1)
        #   M2: (1,1, -1)
        #   M6: (1,1, 1)
        
        # So:
        # M1: (0,0, 1), (1,1, 1)
        # M2: (0,1, 1), (1,1, -1)
        # M3: (1,0, 1)
        # M4: (0,1, 1)
        # M5: (0,0, -1), (1,0, 1)
        # M6: (1,1, 1)
        # M7: (0,0, 1)
    ]
    
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],           # M1
        [(0, 1, 1), (1, 1, -1)],          # M2
        [(1, 0, 1)],                      # M3
        [(0, 1, 1)],                      # M4
        [(0, 0, -1), (1, 0, 1)],          # M5
        [(1, 1, 1)],                      # M6
        [(0, 0, 1)],                      # M7
    ]

    col = 0
    for t in range(num_blocks):
        base_col = 2 * t
        for s in range(7):
            for (ai, aj, c) in U_coeffs[s]:
                U[ai * m + aj][col] = c
            for (bj, bk, c) in V_coeffs[s]:
                V[bj * p + base_col + bk][col] = c
            for (ck, ci, c) in W_coeffs[s]:
                W[(base_col + ck) * n + ci][col] = c
            col += 1

    return {"U": U, "V": V, "W": W}
