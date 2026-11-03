import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # We use 8 Strassen instances
    # Each instance handles a 2x2x2 sub-problem
    # Total rank = 8 * 7 = 56
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A, B, C
    # A is 2x2, B is 2x2, C is 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define the linear combinations
    # A indices: 00, 01, 10, 11
    # B indices: 00, 01, 10, 11
    
    # We iterate over the 8 blocks
    # K: 0, 1 (k major half)
    # J: 0, 1 (j half)
    # K': 0, 1 (k minor half)
    
    col_offset = 0
    for K in range(2):
        for J in range(2):
            for Kp in range(2):
                # Base column for this Strassen instance
                base_col = col_offset
                
                # Map local indices to global
                # A local (i, ja) -> global (i, 2*J + ja)
                # B local (jb, kb) -> global (2*J + jb, 4*K + 2*Kp + kb)
                # C local (i, kb) -> global (4*K + 2*Kp + kb, i)
                
                # 7 multiplications
                for mult_idx in range(7):
                    c = base_col + mult_idx
                    
                    # Determine coefficients for U (A part)
                    # Determine coefficients for V (B part)
                    # Determine coefficients for W (C part)
                    
                    # We define the terms for each M
                    # M1: A: 00,11; B: 00,11
                    # M2: A: 10,11; B: 00
                    # M3: A: 00; B: 10,11
                    # M4: A: 11; B: 01,00
                    # M5: A: 00,10; B: 11
                    # M6: A: 10,00; B: 00,01
                    # M7: A: 01,11; B: 10,11
                    
                    # We need to add 1s or -1s to the appropriate cells in U, V, W
                    
                    # Helper to add to U
                    def add_U(i, ja, val):
                        U[i * m + (2 * J + ja)][c] += val
                    
                    # Helper to add to V
                    def add_V(jb, kb, val):
                        V[(2 * J + jb) * p + (4 * K + 2 * Kp + kb)][c] += val
                    
                    # Helper to add to W
                    def add_W(kb, i, val):
                        W[(4 * K + 2 * Kp + kb) * n + i][c] += val

                    if mult_idx == 0: # M1
                        add_U(0, 0, 1)
                        add_U(1, 1, 1)
                        add_V(0, 0, 1)
                        add_V(1, 1, 1)
                        add_W(0, 0, 1)
                        add_W(1, 1, 1)
                        add_W(0, 1, 1)
                        add_W(1, 0, 1)
                        
                    elif mult_idx == 1: # M2
                        add_U(1, 0, 1)
                        add_U(1, 1, 1)
                        add_V(0, 0, 1)
                        add_W(0, 1, 1)
                        add_W(1, 1, 1)
                        
                    elif mult_idx == 2: # M3
                        add_U(0, 0, 1)
                        add_V(1, 0, 1)
                        add_V(1, 1, -1)
                        add_W(1, 0, 1)
                        add_W(1, 1, 1)
                        
                    elif mult_idx == 3: # M4
                        add_U(1, 1, 1)
                        add_V(0, 1, 1)
                        add_V(0, 0, -1)
                        add_W(0, 0, 1)
                        add_W(1, 1, 1)
                        
                    elif mult_idx == 4: # M5
                        add_U(0, 0, 1)
                        add_U(1, 0, 1)
                        add_V(1, 1, 1)
                        add_W(0, 0, -1)
                        add_W(1, 0, 1)
                        
                    elif mult_idx == 5: # M6
                        add_U(1, 0, 1)
                        add_U(0, 0, -1)
                        add_V(0, 0, 1)
                        add_V(0, 1, 1)
                        add_W(1, 1, 1)
                        
                    elif mult_idx == 6: # M7
                        add_U(0, 1, 1)
                        add_U(1, 1, -1)
                        add_V(1, 0, 1)
                        add_V(1, 1, 1)
                        add_W(0, 0, 1)
                        add_W(1, 0, 1)

                col_offset += 7
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
