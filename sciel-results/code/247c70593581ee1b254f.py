import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen coefficients for 2x2x2
    # U: (i, j) -> coeff for A[i,j]
    # V: (j, ell) -> coeff for B[j,ell]
    # W: (ell, i) -> coeff for C[i,ell]
    # Indices i, j, ell in {0, 1}
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B10 - B11)
    # M4 = A11 (B01 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A11 - A10)(B10 + B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M6
    
    U_s = [
        [1, 0], [0, 0], # M1: A00, A01, A10, A11 -> wait, U is (i,j).
        # Let's use a 4x7 matrix for U_s where index is i*2+j
        # Row 0: i=0,j=0. Row 1: i=0,j=1. Row 2: i=1,j=0. Row 3: i=1,j=1.
    ]
    
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0
        [0, 0, 0, 0, 1, 0, 0],  # i=0, j=1
        [0, 1, 0, 0, 0, 1, -1], # i=1, j=0
        [1, 1, 0, 1, 0, 0, 1]   # i=1, j=1
    ])
    
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0], # j=0, ell=0
        [0, 0, 0, 1, 0, 0, 0],  # j=0, ell=1
        [0, 0, 1, 0, 1, 0, 1],  # j=1, ell=0
        [1, 0, -1, 0, 1, 0, 1]  # j=1, ell=1
    ])
    
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, -1], # ell=0, i=0
        [0, 0, 1, 0, 1, 0, 0],   # ell=0, i=1
        [0, 0, 0, 0, 0, 0, 1],   # ell=1, i=0 -- Wait, C10 = M2 + M7. i=1, ell=0.
        # Let's re-verify W indices.
        # W index is (ell, i).
        # C00 (i=0, ell=0): M1, M4, -M5, -M7.
        # C01 (i=0, ell=1): M3, M5.
        # C10 (i=1, ell=0): M2, M7.
        # C11 (i=1, ell=1): M1, -M2, M3, M6.
        
    ])
    
    # W_s rows:
    # Row 0: ell=0, i=0 -> C00 coeffs: [1, 0, 0, 1, -1, 0, -1]
    # Row 1: ell=0, i=1 -> C10 coeffs: [0, 1, 0, 0, 0, 0, 1]
    # Row 2: ell=1, i=0 -> C01 coeffs: [0, 0, 1, 0, 1, 0, 0]
    # Row 3: ell=1, i=1 -> C11 coeffs: [1, -1, 1, 0, 0, 1, 0]
    
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, -1], # ell=0, i=0
        [0, 1, 0, 0, 0, 0, 1],   # ell=0, i=1
        [0, 0, 1, 0, 1, 0, 0],   # ell=1, i=0
        [1, -1, 1, 0, 0, 1, 0]   # ell=1, i=1
    ])
    
    rank = 7 * (p // 2) # 7 * 8 = 56
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for q in range(p // 2):
        for r in range(7):
            r_global = 7 * q + r
            
            # Fill U
            for i in range(n):
                for j in range(m):
                    U[i * m + j, r_global] = U_s[i * m + j, r]
                    
            # Fill V
            # V index: j * p + k. k = 2*q + ell.
            for j in range(m):
                for ell in range(2):
                    k = 2 * q + ell
                    V[j * p + k, r_global] = V_s[j * 2 + ell, r]
                    
            # Fill W
            # W index: k * n + i. k = 2*q + ell.
            for ell in range(2):
                for i in range(n):
                    k = 2 * q + ell
                    W[k * n + i, r_global] = W_s[ell * n + i, r]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
