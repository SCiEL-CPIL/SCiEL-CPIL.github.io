import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add to U, V, W
    def add_term(U_idx, U_val, V_idx, V_val, W_idx, W_val, c):
        U[U_idx][c] += U_val
        V[V_idx][c] += V_val
        W[W_idx][c] += W_val

    column = 0
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen Terms for A_L (cols 0,1) * B_L (rows 0,1)
        
        # T1: (A00 + A11) * (B00 + B11)
        # U: A00 (idx 0), A11 (idx 4)
        # V: B00 (idx 0*8+k0), B11 (idx 1*8+k1)
        # W: Z00 (k0*2+0), Z01 (k1*2+0), Z10 (k0*2+1), Z11 (k1*2+1)
        add_term(0, 1, 0*8+k0, 1, k0*2+0, 1, column)
        add_term(4, 1, 0*8+k0, 1, k0*2+0, 1, column)
        add_term(0, 1, 1*8+k1, 1, k1*2+0, 1, column)
        add_term(4, 1, 1*8+k1, 1, k1*2+0, 1, column)
        add_term(0, 1, 0*8+k0, 1, k0*2+1, 1, column)
        add_term(4, 1, 0*8+k0, 1, k0*2+1, 1, column)
        add_term(0, 1, 1*8+k1, 1, k1*2+1, 1, column)
        add_term(4, 1, 1*8+k1, 1, k1*2+1, 1, column)
        column += 1
        
        # T2: (A10 + A11) * B00
        # U: A10 (idx 3), A11 (idx 4)
        # V: B00 (idx 0*8+k0)
        # W: Z10 (k0*2+1) coeff 1, Z11 (k1*2+1) coeff -1
        add_term(3, 1, 0*8+k0, 1, k0*2+1, 1, column)
        add_term(4, 1, 0*8+k0, 1, k0*2+1, 1, column)
        add_term(3, 1, 0*8+k0, 1, k1*2+1, -1, column)
        add_term(4, 1, 0*8+k0, 1, k1*2+1, -1, column)
        column += 1
        
        # T3: A00 * (B01 - B11)
        # U: A00 (idx 0)
        # V: B01 (idx 0*8+k1) coeff 1, B11 (idx 1*8+k1) coeff -1
        # W: Z01 (k1*2+0) coeff 1, Z11 (k1*2+1) coeff 1
        add_term(0, 1, 0*8+k1, 1, k1*2+0, 1, column)
        add_term(0, 1, 1*8+k1, -1, k1*2+0, 1, column)
        add_term(0, 1, 0*8+k1, 1, k1*2+1, 1, column)
        add_term(0, 1, 1*8+k1, -1, k1*2+1, 1, column)
        column += 1
        
        # T4: A11 * (B10 - B00)
        # U: A11 (idx 4)
        # V: B10 (idx 1*8+k0) coeff 1, B00 (idx 0*8+k0) coeff -1
        # W: Z10 (k0*2+1) coeff 1, Z11 (k1*2+1) coeff 1
        add_term(4, 1, 1*8+k0, 1, k0*2+1, 1, column)
        add_term(4, 1, 0*8+k0, -1, k0*2+1, 1, column)
        add_term(4, 1, 1*8+k0, 1, k1*2+1, 1, column)
        add_term(4, 1, 0*8+k0, -1, k1*2+1, 1, column)
        column += 1
        
        # T5: (A00 + A01) * B11
        # U: A00 (idx 0), A01 (idx 1)
        # V: B11 (idx 1*8+k1)
        # W: Z00 (k0*2+0) coeff -1, Z01 (k1*2+0) coeff 1
        add_term(0, 1, 1*8+k1, 1, k0*2+0, -1, column)
        add_term(1, 1, 1*8+k1, 1, k0*2+0, -1, column)
        add_term(0, 1, 1*8+k1, 1, k1*2+0, 1, column)
        add_term(1, 1, 1*8+k1, 1, k1*2+0, 1, column)
        column += 1
        
        # T6: (A10 - A00) * (B00 + B01)
        # U: A10 (idx 3) coeff 1, A00 (idx 0) coeff -1
        # V: B00 (idx 0*8+k0) coeff 1, B01 (idx 0*8+k1) coeff 1
        # W: Z11 (k1*2+1) coeff 1
        add_term(3, 1, 0*8+k0, 1, k1*2+1, 1, column)
        add_term(0, -1, 0*8+k0, 1, k1*2+1, 1, column)
        add_term(3, 1, 0*8+k1, 1, k1*2+1, 1, column)
        add_term(0, -1, 0*8+k1, 1, k1*2+1, 1, column)
        column += 1
        
        # T7: (A01 - A11) * (B10 + B11)
        # U: A01 (idx 1) coeff 1, A11 (idx 4) coeff -1
        # V: B10 (idx 1*8+k0) coeff 1, B11 (idx 1*8+k1) coeff 1
        # W: Z00 (k0*2+0) coeff 1
        add_term(1, 1, 1*8+k0, 1, k0*2+0, 1, column)
        add_term(4, -1, 1*8+k0, 1, k0*2+0, 1, column)
        add_term(1, 1, 1*8+k1, 1, k0*2+0, 1, column)
        add_term(4, -1, 1*8+k1, 1, k0*2+0, 1, column)
        column += 1
        
        # Outer Product Terms for A_R (col 2) * B_R (row 2)
        
        # T8: A02 * B20
        add_term(0*3+2, 1, 2*8+k0, 1, k0*2+0, 1, column)
        column += 1
        
        # T9: A02 * B21
        add_term(0*3+2, 1, 2*8+k1, 1, k1*2+0, 1, column)
        column += 1
        
        # T10: A12 * B20
        add_term(1*3+2, 1, 2*8+k0, 1, k0*2+1, 1, column)
        column += 1
        
        # T11: A12 * B21
        add_term(1*3+2, 1, 2*8+k1, 1, k1*2+1, 1, column)
        column += 1

    return {"U": U, "V": V, "W": W}
