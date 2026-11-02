import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # Rank = 6 blocks * 7 Strassen terms = 42
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # Coefficients for A in each M
    # Index: i*m+j -> i*2+j
    U_coeffs = [
        [1, 0, 1, 0], # M0: A00, A01, A10, A11 -> A00=1, A11=1
        [0, 0, 1, 1], # M1: A10=1, A11=1
        [1, 0, 0, 0], # M2: A00=1
        [0, 0, 0, 1], # M3: A11=1
        [1, 1, 0, 0], # M4: A00=1, A01=1
        [-1, 0, 1, 0],# M5: A10=1, A00=-1
        [0, 1, 0, -1] # M6: A01=1, A11=-1
    ]
    
    # Coefficients for B in each M (local 2x2)
    # Index: j*p+k_local -> but we handle global k later
    # For now, local j, k_local
    V_coeffs = [
        [1, 1, 1, 1], # M0: B00=1, B01=1, B10=1, B11=1 (Wait, M0 is (A00+A11)(B00+B11))
                      # B00 coeff 1, B11 coeff 1. B01, B10 are 0.
                      # Let's re-list carefully.
        # M0: (A00+A11)(B00+B11) -> B00:1, B11:1
        # M1: (A10+A11)B00 -> B00:1
        # M2: A00(B01-B11) -> B01:1, B11:-1
        # M3: A11(B10-B00) -> B10:1, B00:-1
        # M4: (A00+A01)B11 -> B11:1
        # M5: (A10-A00)(B00+B01) -> B00:1, B01:1
        # M6: (A01-A11)(B10+B11) -> B10:1, B11:1
        
        [1, 0, 0, 1], # M0: B00, B11
        [1, 0, 0, 0], # M1: B00
        [0, 1, 0, -1],# M2: B01, B11
        [-1, 0, 1, 0],# M3: B00, B10
        [0, 0, 0, 1], # M4: B11
        [1, 1, 0, 0], # M5: B00, B01
        [0, 0, 1, 1]  # M6: B10, B11
    ]
    
    # Coefficients for C reconstruction
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    # Index: i, k_local
    W_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # C00: M0, M3, M4, M6
        [0, 0, 1, 0, 1, 0, 0],  # C01: M2, M4
        [0, 1, 0, 1, 0, 0, 0],  # C10: M1, M3
        [1, -1, 1, 0, 0, 1, 0]  # C11: M0, M1, M2, M5
    ]

    for b in range(6):
        for r in range(7):
            r_global = b * 7 + r
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    u_idx = i * m + j
                    U[u_idx][r_global] = U_coeffs[r][u_idx]
            
            # Fill V
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * b + k_local
                    v_idx = j * p + k_global
                    V[v_idx][r_global] = V_coeffs[r][j * 2 + k_local]
            
            # Fill W
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * b + k_local
                    w_idx = k_global * n + i
                    W[w_idx][r_global] = W_coeffs[i * 2 + k_local][r]

    return {"U": U, "V": V, "W": W}
