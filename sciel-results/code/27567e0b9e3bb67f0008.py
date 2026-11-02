import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank: 32 (for 1x4x8) + 4 * 2 * 7 (for 2x4x8 split into 8 2x2x2 Strassen blocks) = 88
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Row 0 (1 x 4 x 8)
    # i=0
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
    # Part 2: Rows 1-2 (2 x 4 x 8)
    # Split into 4 blocks of width 2 for k
    # Each block is 2 x 4 x 2
    # Split 2 x 4 x 2 into two 2 x 2 x 2 Strassen blocks based on j
    for b in range(4):
        k_start = 2 * b
        for s in range(2):
            j_start = 2 * s
            # i_offset = 1 (rows 1, 2)
            # j_offset = j_start (cols j_start, j_start+1)
            # k_offset = k_start (rows k_start, k_start+1)
            
            i_off = 1
            j_off = j_start
            k_off = k_start
            
            # Strassen 7 multiplications
            # M1 = (A11 + A22) * (B11 + B22)
            # M2 = (A21 + A22) * B11
            # M3 = A11 * (B12 - B22)
            # M4 = A22 * (B21 - B11)
            # M5 = (A11 + A12) * B22
            # M6 = (A21 - A11) * (B11 + B12)
            # M7 = (A12 - A22) * (B21 + B22)
            
            # Indices in local 2x2 block:
            # A: (0,0)->A11, (0,1)->A12, (1,0)->A21, (1,1)->A22
            # B: (0,0)->B11, (0,1)->B12, (1,0)->B21, (1,1)->B22
            # C: (0,0)->C11, (0,1)->C12, (1,0)->C21, (1,1)->C22
            
            # Global indices:
            # A[i, j] -> U[i_off + i, j_off + j]
            # B[j, k] -> V[j_off + j, k_off + k]
            # C[i, k] -> W[k_off + k, i_off + i]
            
            def add_term(u_coeffs, v_coeffs, w_coeffs):
                nonlocal col
                # u_coeffs: dict { (i, j): coeff }
                # v_coeffs: dict { (j, k): coeff }
                # w_coeffs: dict { (i, k): coeff }
                
                for (i, j), val in u_coeffs.items():
                    if val != 0:
                        U[i_off + i][col] += val
                        V[j_off + j][col] += 0 # V is handled by v_coeffs
                # Wait, U and V are separate.
                # U row index is i*m + j.
                # V row index is j*p + k.
                
                # Let's just fill directly.
                
                # Reset col logic? No, we increment col once per term.
                
                # Fill U
                for (i, j), val in u_coeffs.items():
                    if val != 0:
                        U[i_off + i][col] += val
                        
                # Fill V
                for (j, k), val in v_coeffs.items():
                    if val != 0:
                        V[j_off + j][col] += val
                        
                # Fill W
                for (i, k), val in w_coeffs.items():
                    if val != 0:
                        W[k_off + k][col] += val
                        
                col += 1
            
            # M1
            u1 = {(0,0): 1, (1,1): 1}
            v1 = {(0,0): 1, (1,1): 1}
            w1 = {(0,0): 1, (1,1): 1}
            add_term(u1, v1, w1)
            
            # M2
            u2 = {(1,0): 1, (1,1): 1}
            v2 = {(0,0): 1}
            w2 = {(1,0): 1, (1,1): -1} # C21 += M2, C22 -= M2
            add_term(u2, v2, w2)
            
            # M3
            u3 = {(0,0): 1}
            v3 = {(0,1): 1, (1,1): -1}
            w3 = {(0,1): 1, (1,1): 1} # C12 += M3, C22 += M3
            add_term(u3, v3, w3)
            
            # M4
            u4 = {(1,1): 1}
            v4 = {(1,0): 1, (0,0): -1}
            w4 = {(0,0): 1, (1,0): 1} # C11 += M4, C21 += M4
            add_term(u4, v4, w4)
            
            # M5
            u5 = {(0,0): 1, (0,1): 1}
            v5 = {(1,1): 1}
            w5 = {(0,0): -1, (0,1): 1} # C11 -= M5, C12 += M5
            add_term(u5, v5, w5)
            
            # M6
            u6 = {(1,0): 1, (0,0): -1}
            v6 = {(0,0): 1, (0,1): 1}
            w6 = {(1,1): 1} # C22 += M6
            add_term(u6, v6, w6)
            
            # M7
            u7 = {(0,1): 1, (1,1): -1}
            v7 = {(1,0): 1, (1,1): 1}
            w7 = {(0,0): 1} # C11 += M7
            add_term(u7, v7, w7)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
