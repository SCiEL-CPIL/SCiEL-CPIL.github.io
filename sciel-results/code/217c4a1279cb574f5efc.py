import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen coefficients for 2x2 blocks
    # M_k = X_k * Y_k
    # X_k, Y_k are 2x2 matrices of blocks
    
    # Outer Strassen definitions
    # k: 0..6
    # X_k coeffs for A_blocks (4 blocks: 00, 01, 10, 11)
    # Y_k coeffs for B_blocks (4 blocks: 00, 01, 10, 11)
    
    # A_blocks indices: 0:00, 1:01, 2:10, 3:11
    # B_blocks indices: 0:00, 1:01, 2:10, 3:11
    
    outer_X = [
        [1, 0, 0, 0], # M1 = A00
        [1, 1, 0, 0], # M2 = A00+A01
        [1, 0, 0, 0], # M3 = A00
        [0, 0, 0, 1], # M4 = A11
        [-1, 0, 1, 0],# M5 = A10-A00
        [0, 1, 0, 1], # M6 = A01+A11
        [-1, 0, 1, -1]# M7 = A10-A11
    ]
    
    outer_Y = [
        [1, 0, 0, 0], # M1 = B00
        [0, 0, 1, 0], # M2 = B10
        [0, -1, 1, 0],# M3 = B10-B11
        [1, 0, 0, 1], # M4 = B00+B11
        [0, 1, 0, 0], # M5 = B01
        [0, -1, 1, 0],# M6 = B10-B11
        [1, 1, 0, 0]  # M7 = B00+B01
    ]
    
    # Inner Strassen definitions for 2x2 matrices
    # P_l = Lx_l * Ly_l
    # l: 0..6
    # Lx_l coeffs for x_elements (4: 00, 01, 10, 11)
    # Ly_l coeffs for y_elements (4: 00, 01, 10, 11)
    
    inner_Lx = [
        [1, 0, 0, 0], # P1 = x00
        [0, 0, 0, 1], # P2 = x11
        [1, 1, 0, 0], # P3 = x00+x01
        [0, 0, 0, 1], # P4 = x11
        [1, 0, 0, 0], # P5 = x00
        [0, 0, 1, -1],# P6 = x10-x11
        [0, 0, 1, 0]  # P7 = x10
    ]
    
    inner_Ly = [
        [1, 0, 0, 0], # P1 = y00
        [0, 0, 0, 1], # P2 = y11
        [0, 0, 1, 0], # P3 = y10
        [0, 0, 1, -1],# P4 = y10-y11
        [0, 1, 0, 1], # P5 = y01+y11
        [1, 0, 0, 0], # P6 = y00
        [1, -1, 0, 0] # P7 = y00-y11
    ]
    
    # Contribution of P_l to Z elements (4: 00, 01, 10, 11)
    # Z_u = sum c_l P_l
    inner_Z = [
        [1, 0, 0, 1, 0, 0, -1], # Z00 = P1+P2+P4-P7
        [0, 0, 0, 0, 1, 0, 1],  # Z01 = P5+P7
        [0, 0, 1, 0, 0, 0, 0],  # Z10 = P3
        [1, 0, -1, 0, 1, -1, 0] # Z11 = P1-P3+P5-P6
    ]
    # Wait, let's re-verify Z11.
    # Z11 = P1 + P5 - P3 - P6.
    # My list: [1, 0, -1, 0, 1, -1, 0]. Correct.
    # Z10 = P3 + P2.
    # My list: [0, 0, 1, 0, 0, 0, 0]. Missing P2!
    # Z10 = P3 + P2.
    # So inner_Z[2] should be [0, 1, 1, 0, 0, 0, 0].
    
    inner_Z[2] = [0, 1, 1, 0, 0, 0, 0]
    
    # Contribution of M_k to C_blocks (4: 00, 01, 10, 11)
    # C_ab = sum c_k M_k
    outer_C = [
        [1, 0, 0, 1, 0, -1, -1], # C00 = M1+M4-M6-M7
        [1, 0, 0, 0, 1, 0, 0],   # C01 = M1+M5
        [0, 1, 0, 1, 0, 0, 0],   # C10 = M2+M4
        [-1, 1, 1, 0, 0, 0, 1]   # C11 = -M1+M2+M3+M7
    ]
    # Check C11: M3+M2-M1+M7.
    # Coeffs: M1:-1, M2:1, M3:1, M4:0, M5:0, M6:0, M7:1.
    # My list: [-1, 1, 1, 0, 0, 0, 1]. Correct.

    for k in range(7):
        for l in range(7):
            r = k * 7 + l
            
            # 1. Construct U and V
            # U[i*m+j, r]
            # V[j*p+k, r]
            
            # L_A is linear combo of A_blocks
            # A_block_ab consists of A[2a+u, 2b+v]
            
            # Coeff of A[2a+u, 2b+v] in X_k
            # X_k = sum_ab outer_X[k][ab_idx] * A_block_ab
            # Element (u,v) of X_k is sum_ab outer_X[k][ab_idx] * A[2a+u, 2b+v]
            
            # Lx_l is linear combo of X_k elements
            # Lx_l = sum_uv inner_Lx[l][uv_idx] * X_k[u,v]
            
            # So coeff of A[2a+u, 2b+v] in Lx_l is:
            # sum_ab outer_X[k][ab_idx] * inner_Lx[l][uv_idx]
            # where ab_idx = a*2+b, uv_idx = u*2+v
            
            for a in range(2):
                for b in range(2):
                    ab_idx = a * 2 + b
                    coeff_block = outer_X[k][ab_idx]
                    if coeff_block == 0: continue
                    
                    for u in range(2):
                        for v in range(2):
                            uv_idx = u * 2 + v
                            coeff_elem = inner_Lx[l][uv_idx]
                            if coeff_elem == 0: continue
                            
                            coeff_total = coeff_block * coeff_elem
                            if coeff_total != 0:
                                i = 2 * a + u
                                j = 2 * b + v
                                U[i * m + j][r] += coeff_total
            
            # Similarly for V
            for c in range(2):
                for d in range(2):
                    cd_idx = c * 2 + d
                    coeff_block = outer_Y[k][cd_idx]
                    if coeff_block == 0: continue
                    
                    for u in range(2):
                        for v in range(2):
                            uv_idx = u * 2 + v
                            coeff_elem = inner_Ly[l][uv_idx]
                            if coeff_elem == 0: continue
                            
                            coeff_total = coeff_block * coeff_elem
                            if coeff_total != 0:
                                j = 2 * c + u
                                k_out = 2 * d + v # This is the 'k' index of B, which is the output col of B
                                # B[j, k_out]
                                V[j * p + k_out][r] += coeff_total
            
            # 2. Construct W
            # W[k_out*n+i, r]
            
            # Coeff of C[i, k_out]
            # C[i, k_out] is in block C_ab where a=i//2, b=k_out//2
            # Position in block is (u, v) where u=i%2, v=k_out%2
            
            # Z_k[u, v] contributes to C_ab with coeff outer_C[ab_idx][k]
            # P_l contributes to Z_k[u, v] with coeff inner_Z[uv_idx][l]
            
            for a in range(2):
                for b in range(2):
                    ab_idx = a * 2 + b
                    coeff_block_C = outer_C[ab_idx][k]
                    if coeff_block_C == 0: continue
                    
                    for u in range(2):
                        for v in range(2):
                            uv_idx = u * 2 + v
                            coeff_elem_Z = inner_Z[uv_idx][l]
                            if coeff_elem_Z == 0: continue
                            
                            coeff_total = coeff_block_C * coeff_elem_Z
                            if coeff_total != 0:
                                i = 2 * a + u
                                k_out = 2 * b + v
                                W[k_out * n + i][r] += coeff_total

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
