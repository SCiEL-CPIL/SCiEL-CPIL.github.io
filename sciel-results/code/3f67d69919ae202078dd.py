import time

_START = time.time()
_DEADLINE = _START + 110.0


def _deadline_exceeded():
    return time.time() > _DEADLINE


def _build_rank21():
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2 multiplication
    # U (Left factor A) indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # V (Right factor B) indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (local)
    # W (Result C) indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (local)
    
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A10)(B10+B11)
    
    U_coeffs = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {2: 1, 0: -1},      # M6
        {1: 1, 2: -1}       # M7
    ]
    
    # V_coeffs[k] is a list of (local_j, local_k, val)
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)],  # M1: B00+B11
        [(0, 0, 1)],             # M2: B00
        [(1, 0, 1), (1, 1, -1)], # M3: B10-B11
        [(0, 1, 1), (0, 0, -1)], # M4: B01-B00
        [(1, 1, 1)],             # M5: B11
        [(0, 0, 1), (0, 1, 1)],  # M6: B00+B01
        [(1, 0, 1), (1, 1, 1)]   # M7: B10+B11
    ]
    
    # W_coeffs[k] is a list of (local_k, local_i, val)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M4
    
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],  # M1: C00, C11
        [(1, 0, 1), (1, 1, -1)], # M2: C10, -C11
        [(1, 0, 1)],             # M3: C11 (Wait, M3 is only in C11? No, C11 = M1-M2+M3+M4)
        [(0, 0, 1), (1, 1, 1)],  # M4: C00, C11
        [(0, 0, -1), (0, 1, 1)], # M5: -C00, C01
        [(1, 0, 1)],             # M6: C10
        [(0, 0, -1), (0, 1, 1)]  # M7: -C00, C01
    ]
    
    # Let's re-verify W_coeffs carefully.
    # C00 = M1 + M4 - M5 - M7
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M4
    
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],  # M1
        [(1, 0, 1), (1, 1, -1)], # M2
        [(1, 1, 1)],             # M3
        [(0, 0, 1), (1, 1, 1)],  # M4
        [(0, 0, -1), (0, 1, 1)], # M5
        [(1, 0, 1)],             # M6
        [(0, 0, -1), (0, 1, 1)]  # M7
    ]
    
    for r in range(3):
        for k in range(7):
            col = r * 7 + k
            for idx, val in U_coeffs[k].items():
                U[idx][col] = val
            for j_local, k_local, val in V_coeffs[k]:
                global_k = 2 * r + k_local
                V[j_local * p + global_k][col] = val
            for i, j_local, val in W_coeffs[k]:
                global_k = 2 * r + j_local
                W[global_k * n + i][col] = val
    return {"U": U, "V": V, "W": W}


def construct():
    """Return {'U','V','W'} factor matrices exactly decomposing the 2,2,6 MM tensor."""
    return _build_rank21()
