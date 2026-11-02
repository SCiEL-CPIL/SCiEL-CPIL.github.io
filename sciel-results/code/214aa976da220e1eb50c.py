# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0  # Safety margin before 120s limit

def _time_remaining():
    return _DEADLINE - time.time()

def _build_rank25():
    """
    Build the known-good rank-25 decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for columns 0,1
    - Use Strassen's rank-7 algorithm for columns 2,3
    - Use Strassen's rank-7 algorithm for columns 4,5
    - Use schoolbook (rank 4) for column 6
    Total rank: 7 + 7 + 7 + 4 = 25
    """
    n, m, p = 2, 2, 7
    rank = 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 * 2x2
    # A indices: 0:A00, 1:A01, 2:A10, 3:A11
    strassen_U = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [-1, 0, 1, 0],# M6: -A00 + A10
        [0, 1, 0, -1] # M7: A01 - A11
    ]
    
    # W (C) coeffs for M1..M7
    # Local C indices: 0:C00, 1:C01, 2:C10, 3:C11
    strassen_W_local = [
        {0: 1, 3: 1},     # M1: C00 + C11
        {2: 1, 3: -1},    # M2: C10 - C11
        {1: 1, 3: 1},     # M3: C01 + C11
        {0: 1, 2: 1},     # M4: C00 + C10
        {0: -1, 1: 1},    # M5: -C00 + C01
        {3: 1},           # M6: C11
        {0: 1}            # M7: C00
    ]
    
    # V (B) coeffs for M1..M7
    # Local B indices: 0:B00, 1:B01, 2:B10, 3:B11
    strassen_V_local = [
        {0: 1, 3: 1},     # M1: B00 + B11
        {0: 1},           # M2: B00
        {1: 1, 3: -1},    # M3: B01 - B11
        {2: 1, 0: -1},    # M4: B10 - B00
        {3: 1},           # M5: B11
        {0: 1, 1: 1},     # M6: B00 + B01
        {2: 1, 3: 1}      # M7: B10 + B11
    ]
    
    col = 0
    
    # Process 3 Strassen blocks (columns 0-5)
    for b in range(3):
        for r in range(7):
            # U
            for i in range(4):
                U[i][col] += strassen_U[r][i]
            
            # V
            for loc_idx, coeff in strassen_V_local[r].items():
                if loc_idx == 0:
                    v_idx = 0 * p + (2 * b)
                elif loc_idx == 1:
                    v_idx = 0 * p + (2 * b + 1)
                elif loc_idx == 2:
                    v_idx = 1 * p + (2 * b)
                elif loc_idx == 3:
                    v_idx = 1 * p + (2 * b + 1)
                V[v_idx][col] += coeff
            
            # W
            for loc_idx, coeff in strassen_W_local[r].items():
                if loc_idx == 0:
                    w_idx = (2 * b) * n + 0
                elif loc_idx == 1:
                    w_idx = (2 * b + 1) * n + 0
                elif loc_idx == 2:
                    w_idx = (2 * b) * n + 1
                elif loc_idx == 3:
                    w_idx = (2 * b + 1) * n + 1
                W[w_idx][col] += coeff
            
            col += 1
            
    # Schoolbook for last block (column 6)
    # S1: A00 * B06 -> C06
    # S2: A01 * B16 -> C06
    # S3: A10 * B06 -> C16
    # S4: A11 * B16 -> C16
    
    # S1
    U[0][col] += 1
    V[0 * p + 6][col] += 1
    W[6 * n + 0][col] += 1
    col += 1
    
    # S2
    U[1][col] += 1
    V[1 * p + 6][col] += 1
    W[6 * n + 0][col] += 1
    col += 1
    
    # S3
    U[2][col] += 1
    V[0 * p + 6][col] += 1
    W[6 * n + 1][col] += 1
    col += 1
    
    # S4
    U[3][col] += 1
    V[1 * p + 6][col] += 1
    W[6 * n + 1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

def _build_rank28_fallback():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][idx] = 1
                V[j*p + k][idx] = 1
                W[k*n + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Construct a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Returns dict with keys 'U', 'V', 'W' as lists of lists with integer entries.
    """
    if _time_remaining() < 2.0:
        # Safety: return schoolbook if we're about to time out
        return _build_rank28_fallback()
    
    result = _build_rank25()
    
    # Quick verification to ensure correctness
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    n, m, p = 2, 2, 7
    rank = U.shape[1]
    
    # Verify with multiple random tests
    for seed in [42, 123]:
        np.random.seed(seed)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            SA = np.dot(U[:, r], A.flatten())
            SB = np.dot(V[:, r], B.flatten())
            for k in range(p):
                for i in range(n):
                    C_computed[i, k] += W[k*n + i, r] * SA * SB
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # If verification fails, return fallback
            return _build_rank28_fallback()
    
    return result

# EVOLVE-BLOCK-END
