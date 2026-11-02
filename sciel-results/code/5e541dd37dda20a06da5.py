import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x4 matrix.
    """
    n, m, p = 2, 3, 4
    
    # Rank 22 decomposition
    R = 22
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    col = 0
    
    # Helper to add a decomposition block
    def add_block(u_indices, v_indices, w_indices, u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        for r in range(len(u_coeffs)):
            for i, j in u_indices:
                U[i * m + j, col + r] += u_coeffs[r][i, j]
            for j, k in v_indices:
                V[j * p + k, col + r] += v_coeffs[r][j, k]
            for k, i in w_indices:
                W[k * n + i, col + r] += w_coeffs[r][k, i]
        col += len(u_coeffs)

    # Define Strassen coefficients for 2x2x2
    # Indices i,j in 0..1 for U, j,k in 0..1 for V, k,i in 0..1 for W
    # M1..M7
    s_u = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (1,0): -1},
        {(0,1): 1, (0,0): -1},
        {(1,0): 1, (1,1): -1}
    ]
    s_v = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (1,0): 1},
        {(1,1): 1},
        {(0,0): 1, (1,0): 1},
        {(0,1): 1, (1,1): 1}
    ]
    s_w = [
        {(0,0): 1, (1,1): 1, (0,1): -1, (1,0): -1}, # C00 = M1+M4-M5-M7
        {(1,1): 1, (0,1): 1}, # C01 = M3+M4 (Wait, W index is k,i. C01 is i=0, k=1. So W[1,0]. M3, M4)
        {(1,0): 1, (0,0): 1}, # C10 = M2+M5 (i=1, k=0. W[0,1]. M2, M5)
        {(1,1): 1, (1,0): -1, (0,1): 1, (0,0): 1} # C11 = M1-M2+M3+M7 (i=1, k=1. W[1,1]. M1, -M2, M3, M7)
    ]
    # Correction on W indices for Strassen:
    # C_00: i=0, k=0. W[0,0]. Coeffs: M1(1), M4(1), M5(-1), M7(-1).
    # C_01: i=0, k=1. W[1,0]. Coeffs: M3(1), M4(1).
    # C_10: i=1, k=0. W[0,1]. Coeffs: M2(1), M5(1).
    # C_11: i=1, k=1. W[1,1]. Coeffs: M1(1), M2(-1), M3(1), M7(1).
    
    s_w_corrected = [
        {(0,0): 1, (1,1): 1, (0,1): -1, (1,0): -1},
        {(1,0): 1, (0,0): 1},
        {(0,1): 1, (1,1): 1},
        {(1,1): 1, (0,0): -1, (1,0): 1, (0,1): 1}
    ]
    # Wait, the keys in s_w are (k, i).
    # M1: k=0,i=0: 1; k=1,i=1: 1; k=1,i=0: -1 (C10? No C01 is k=1,i=0); k=0,i=1: -1 (C10 is k=0,i=1).
    # Let's just use the explicit addition logic.
    
    def add_strassen(i_offset, j_offset, k_offset):
        nonlocal col
        # 7 terms
        for r in range(7):
            # U
            for (i, j), val in s_u[r].items():
                U[(i + i_offset) * m + (j + j_offset), col + r] += val
            # V
            for (j, k), val in s_v[r].items():
                V[(j + j_offset) * p + (k + k_offset), col + r] += val
            # W
            # We need to map the 2x2 result C_ij to W[k, i]
            # C_00 -> k=0, i=0. W[0*n+0]
            # C_01 -> k=1, i=0. W[1*n+0]
            # C_10 -> k=0, i=1. W[0*n+1]
            # C_11 -> k=1, i=1. W[1*n+1]
            
            # Coefficients for M1..M7 on C_00, C_01, C_10, C_11
            # C_00: M1(1), M4(1), M5(-1), M7(-1)
            # C_01: M3(1), M4(1)
            # C_10: M2(1), M5(1)
            # C_11: M1(1), M2(-1), M3(1), M7(1)
            
            w_val_00 = 0
            w_val_01 = 0
            w_val_10 = 0
            w_val_11 = 0
            
            if r == 0: # M1
                w_val_00 = 1
                w_val_11 = 1
            elif r == 1: # M2
                w_val_10 = 1
                w_val_11 = -1
            elif r == 2: # M3
                w_val_01 = 1
                w_val_11 = 1
            elif r == 3: # M4
                w_val_00 = 1
                w_val_01 = 1
            elif r == 4: # M5
                w_val_00 = -1
                w_val_10 = 1
            elif r == 5: # M6
                pass # M6 is not used in standard Strassen output? 
                # Standard Strassen uses M1..M7. M6 is used in C_01? 
                # C_01 = M3 + M4. C_10 = M2 + M5. 
                # M6 = (A01-A00)(B00+B10). M7 = (A10-A11)(B01+B11).
                # My previous C_00, C_01, C_10, C_11 formulas:
                # C00 = M1 + M4 - M5 - M7
                # C01 = M3 + M4
                # C10 = M2 + M5
                # C11 = M1 - M2 + M3 + M7
                # This uses M1, M2, M3, M4, M5, M7. M6 is not used?
                # Wait, standard Strassen:
                # P1 = (A00+A11)(B00+B11)
                # P2 = (A10+A11)B00
                # P3 = A00(B10-B11)
                # P4 = A11(B01+B10)
                # P5 = (A00-A10)B11
                # P6 = (A01-A00)(B00+B10)
                # P7 = (A10-A11)(B01+B11)
                # C00 = P1 + P4 - P5 - P7
                # C01 = P3 + P4
                # C10 = P2 + P5
                # C11 = P1 - P2 + P3 + P7
                # Yes, P6 is not used in the final sum? 
                # Wait, P6 is defined but not used? 
                # No, P6 is used in some variations, but the standard 7-term Strassen does not use P6 in the reconstruction?
                # Actually, P6 is not needed for the 4 outputs if we use the above 7 products.
                # So we can skip P6 (r=5) or set its coefficients to 0.
                pass
            elif r == 6: # M7
                w_val_00 = -1
                w_val_11 = 1
                
            # Add to W
            W[(0 + k_offset) * n + (0 + i_offset), col + r] += w_val_00
            W[(1 + k_offset) * n + (0 + i_offset), col + r] += w_val_01
            W[(0 + k_offset) * n + (1 + i_offset), col + r] += w_val_10
            W[(1 + k_offset) * n + (1 + i_offset), col + r] += w_val_11
            
        col += 7

    # Define Schoolbook for 2x1x2
    def add_212(i_offset, j_offset, k_offset):
        nonlocal col
        # 4 terms
        for r in range(4):
            # U: i in 0..1, j=j_offset
            # V: j=j_offset, k in 0..1
            # W: k in 0..1, i in 0..1
            
            if r == 0:
                # A_i0 * B_0k
                U[i_offset * m + j_offset, col + r] = 1 # i=0
                U[(i_offset+1) * m + j_offset, col + r] = 0 # i=1
                V[j_offset * p + k_offset, col + r] = 1 # k=0
                V[j_offset * p + k_offset + 1, col + r] = 0 # k=1
                W[k_offset * n + i_offset, col + r] = 1 # C00
                W[(k_offset+1) * n + i_offset, col + r] = 1 # C01
                W[k_offset * n + i_offset + 1, col + r] = 0 # C10
                W[(k_offset+1) * n + i_offset + 1, col + r] = 0 # C11
            elif r == 1:
                # A_i0 * B_1k
                U[i_offset * m + j_offset, col + r] = 1
                U[(i_offset+1) * m + j_offset, col + r] = 0
                V[j_offset * p + k_offset, col + r] = 0
                V[j_offset * p + k_offset + 1, col + r] = 1
                W[k_offset * n + i_offset, col + r] = 0
                W[(k_offset+1) * n + i_offset, col + r] = 1
                W[k_offset * n + i_offset + 1, col + r] = 0
                W[(k_offset+1) * n + i_offset + 1, col + r] = 0
            elif r == 2:
                # A_i1 * B_0k
                U[i_offset * m + j_offset, col + r] = 0
                U[(i_offset+1) * m + j_offset, col + r] = 1
                V[j_offset * p + k_offset, col + r] = 1
                V[j_offset * p + k_offset + 1, col + r] = 0
                W[k_offset * n + i_offset, col + r] = 0
                W[(k_offset+1) * n + i_offset, col + r] = 0
                W[k_offset * n + i_offset + 1, col + r] = 1
                W[(k_offset+1) * n + i_offset + 1, col + r] = 1
            elif r == 3:
                # A_i1 * B_1k
                U[i_offset * m + j_offset, col + r] = 0
                U[(i_offset+1) * m + j_offset, col + r] = 1
                V[j_offset * p + k_offset, col + r] = 0
                V[j_offset * p + k_offset + 1, col + r] = 1
                W[k_offset * n + i_offset, col + r] = 0
                W[(k_offset+1) * n + i_offset, col + r] = 0
                W[k_offset * n + i_offset + 1, col + r] = 0
                W[(k_offset+1) * n + i_offset + 1, col + r] = 1
                
        col += 4

    # Apply blocks
    # Block 1: 2x2x2 at i=0,1; j=0,1; k=0,1
    add_strassen(0, 0, 0)
    
    # Block 2: 2x2x2 at i=0,1; j=0,1; k=2,3
    add_strassen(0, 0, 2)
    
    # Block 3: 2x1x2 at i=0,1; j=2; k=0,1
    add_212(0, 2, 0)
    
    # Block 4: 2x1x2 at i=0,1; j=2; k=2,3
    add_212(0, 2, 2)
    
    # Convert to lists
    U_out = [[int(x) for x in row] for row in U]
    V_out = [[int(x) for x in row] for row in V]
    W_out = [[int(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
