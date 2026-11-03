import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # X, Y, Z are 2x2 matrices
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y10-y11)
    # M4 = x11(y01+y11)
    # M5 = (x00-x10)(y00+y10)
    # M6 = (x01-x11)(y10+y11)
    # M7 = x01(y01-y10)
    
    # Z00 = M1 + M4 - M5 - M7
    # Z01 = M3 + M7
    # Z10 = M2 + M5
    # Z11 = M1 - M2 + M3 + M6
    
    def add_strassen(i_start, j_start, k_start, col_offset):
        # i_start: row start in A (0 or 1? No, A is 2x4. i is 0..1. j is 0..3)
        # Wait, the sub-problems are 2x2.
        # A is 2x4.
        # The "X" matrix in Strassen is 2x2.
        # It comes from A.
        # A rows are 0,1. Cols are j_start, j_start+1.
        # So i in X corresponds to i in A (0,1).
        # j in X corresponds to j in A (j_start, j_start+1).
        
        # B is 4x8.
        # Y is 2x2.
        # Rows are j_start, j_start+1.
        # Cols are k_start, k_start+1.
        
        # C is 8x2.
        # Z is 2x2.
        # Rows are k_start, k_start+1.
        # Cols are i (0,1).
        
        # We need to map the 7 Strassen terms to U, V, W columns.
        
        # Term definitions: (coeff_x, coeff_y, coeff_z)
        # x indices: (i, j) in 2x2 local
        # y indices: (j, k) in 2x2 local
        # z indices: (k, i) in 2x2 local (output)
        
        # We will iterate over the 7 terms.
        
        # M1
        # x: (0,0) + (1,1)
        # y: (0,0) + (1,1)
        # z: (0,0) +1, (1,1) +1
        
        # M2
        # x: (1,0) + (1,1)
        # y: (0,0)
        # z: (1,0) +1, (1,1) -1
        
        # M3
        # x: (0,0)
        # y: (1,0) - (1,1)
        # z: (0,1) +1, (1,1) +1
        
        # M4
        # x: (1,1)
        # y: (0,1) + (1,1)
        # z: (0,0) +1
        
        # M5
        # x: (0,0) - (1,0)
        # y: (0,0) + (1,0)
        # z: (0,0) -1, (1,0) +1
        
        # M6
        # x: (0,1) - (1,1)
        # y: (1,0) + (1,1)
        # z: (1,1) +1
        
        # M7
        # x: (0,1)
        # y: (0,1) - (1,0)
        # z: (0,0) -1, (0,1) +1
        
        terms = [
            # (x_coeffs, y_coeffs, z_coeffs)
            # x_coeffs: list of (i, j, coeff)
            # y_coeffs: list of (j, k, coeff)
            # z_coeffs: list of (k, i, coeff)
            
            # M1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # M2
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            # M3
            ([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            # M4
            ([(1,1,1)], [(0,1,1), (1,1,1)], [(0,0,1)]),
            # M5
            ([(0,0,1), (1,0,-1)], [(0,0,1), (1,0,1)], [(0,0,-1), (1,0,1)]),
            # M6
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(1,1,1)]),
            # M7
            ([(0,1,1)], [(0,1,1), (1,0,-1)], [(0,0,-1), (0,1,1)])
        ]
        
        for t_idx, (x_c, y_c, z_c) in enumerate(terms):
            col = col_offset + t_idx
            
            for (i, j, c) in x_c:
                a_idx = i * m + (j_start + j)
                U[a_idx][col] += c
                
            for (j, k, c) in y_c:
                b_idx = (j_start + j) * p + (k_start + k)
                V[b_idx][col] += c
                
            for (k, i, c) in z_c:
                c_idx = (k_start + k) * n + i
                W[c_idx][col] += c

    # 8 Strassen instances
    # C_L (k 0..3)
    # A_L (j 0..1) * B_LT (j 0..1, k 0..1) -> Z (k 0..1, i 0..1)
    add_strassen(0, 0, 0, 0)
    # A_L (j 0..1) * B_LT (j 0..1, k 2..3) -> Z (k 2..3, i 0..1)
    add_strassen(0, 0, 2, 7)
    # A_R (j 2..3) * B_LB (j 2..3, k 0..1) -> Z (k 0..1, i 0..1)
    add_strassen(0, 2, 0, 14)
    # A_R (j 2..3) * B_LB (j 2..3, k 2..3) -> Z (k 2..3, i 0..1)
    add_strassen(0, 2, 2, 21)
    
    # C_R (k 4..7)
    # A_L (j 0..1) * B_RT (j 0..1, k 4..5) -> Z (k 4..5, i 0..1)
    add_strassen(0, 0, 4, 28)
    # A_L (j 0..1) * B_RT (j 0..1, k 6..7) -> Z (k 6..7, i 0..1)
    add_strassen(0, 0, 6, 35)
    # A_R (j 2..3) * B_RB (j 2..3, k 4..5) -> Z (k 4..5, i 0..1)
    add_strassen(0, 2, 4, 42)
    # A_R (j 2..3) * B_RB (j 2..3, k 6..7) -> Z (k 6..7, i 0..1)
    add_strassen(0, 2, 6, 49)
    
    return {"U": U, "V": V, "W": W}
