"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Indices for 2x2:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (index k*2+i)
    
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the linear forms for U (A-part) and V (B-part) for each of the 7 products.
    # Then we define the W coefficients (C-part) for each output element.
    
    # U[c] corresponds to the coefficient vector for A in product c.
    # V[c] corresponds to the coefficient vector for B in product c.
    # W[c] corresponds to the coefficient vector for C in product c.
    
    # Product 1: (A00+A11) * (B00+B11)
    U[0, 0] = 1; U[3, 0] = 1  # A00, A11
    V[0, 0] = 1; V[3, 0] = 1  # B00, B11
    W[0, 0] = 1; W[3, 0] = 1  # C00, C11
    
    # Product 2: (A10+A11) * B00
    U[2, 1] = 1; U[3, 1] = 1  # A10, A11
    V[0, 1] = 1               # B00
    W[2, 1] = 1; W[3, 1] = -1 # C10, C11
    
    # Product 3: A00 * (B01-B11)
    U[0, 2] = 1               # A00
    V[1, 2] = 1; V[3, 2] = -1 # B01, B11
    W[1, 2] = 1; W[3, 2] = 1  # C01, C11
    
    # Product 4: A11 * (B10-B00)
    U[3, 3] = 1               # A11
    V[2, 3] = 1; V[0, 3] = -1 # B10, B00
    W[0, 3] = 1; W[2, 3] = 1  # C00, C10
    
    # Product 5: (A00+A01) * B11
    U[0, 4] = 1; U[1, 4] = 1  # A00, A01
    V[3, 4] = 1               # B11
    W[0, 4] = -1; W[1, 4] = 1 # C00, C01
    
    # Product 6: (A10-A00) * (B00+B01)
    U[2, 5] = 1; U[0, 5] = -1 # A10, A00
    V[0, 5] = 1; V[1, 5] = 1  # B00, B01
    W[3, 5] = 1               # C11
    
    # Product 7: (A01-A11) * (B10+B11)
    U[1, 6] = 1; U[3, 6] = -1 # A01, A11
    V[2, 6] = 1; V[3, 6] = 1  # B10, B11
    W[0, 6] = 1               # C00
    
    return U, V, W

def _compose_4x4_from_2x2(U2, V2, W2):
    """
    Composes a 4x4x4 decomposition from a 2x2x2 decomposition.
    Returns U, V, W for 4x4x4 with rank 49.
    """
    rank2 = U2.shape[1]
    rank4 = rank2 * rank2
    U4 = np.zeros((16, rank4))
    V4 = np.zeros((16, rank4))
    W4 = np.zeros((16, rank4))
    
    # Strassen 4x4 structure:
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    # C blocks: C00, C01, C10, C11 (each 2x2)
    
    # The 7 super-products M1..M7 are 2x2 matrix multiplications.
    # For each super-product r, we use the 2x2 Strassen decomposition (s=0..6).
    
    # Define the linear combinations for X_r (A-part) and Y_r (B-part)
    # X_r is a 2x2 matrix formed from A blocks.
    # Y_r is a 2x2 matrix formed from B blocks.
    
    # We need to map the local 2x2 indices of X_r, Y_r to the global 4x4 indices.
    # Global A index: i*4 + j.
    # Block A_ab has local indices (i,j) -> global (a*2+i, b*2+j).
    
    # For each r in 0..6:
    #   X_r[i, j] = sum_{a,b} coeff_A[r][a,b][i,j] * A_{a*2+i, b*2+j}
    #   Y_r[i, j] = sum_{a,b} coeff_B[r][a,b][i,j] * B_{a*2+i, b*2+j}
    
    # Let's define the coefficients for X_r and Y_r blocks.
    # X_r is a 2x2 matrix. Let's say X_r = sum_{a,b} x_{r,a,b} A_{ab}.
    # Actually, X_r is formed by adding/subtracting A blocks.
    # M1: X = A00+A11, Y = B00+B11
    # M2: X = A10+A11, Y = B00
    # M3: X = A00, Y = B01-B11
    # M4: X = A11, Y = B10-B00
    # M5: X = A00+A01, Y = B11
    # M6: X = A10-A00, Y = B00+B01
    # M7: X = A01-A11, Y = B10+B11
    
    # coeff_A[r][a][b] is the coefficient of block A_{ab} in X_r.
    # Note: X_r is a 2x2 matrix. The "blocks" A_{ab} are 2x2 matrices.
    # So X_r is a linear combination of the 2x2 blocks A_{ab}.
    # Thus, for each local index (i,j) in the 2x2 X_r, the value is
    # sum_{a,b} coeff_A[r][a][b] * A_{a*2+i, b*2+j}.
    
    coeff_A = [
        [[1, 0], [0, 1]],   # M1: A00+A11
        [[0, 0], [1, 1]],   # M2: A10+A11
        [[1, 0], [0, 0]],   # M3: A00
        [[0, 0], [0, 1]],   # M4: A11
        [[1, 1], [0, 0]],   # M5: A00+A01
        [[-1, 0], [1, 0]],  # M6: A10-A00
        [[0, -1], [0, 1]]   # M7: A01-A11
    ]
    
    coeff_B = [
        [[1, 0], [0, 1]],   # M1: B00+B11
        [[1, 0], [0, 0]],   # M2: B00
        [[0, 1], [0, -1]],  # M3: B01-B11
        [[-1, 0], [1, 0]],  # M4: B10-B00
        [[0, 0], [0, 1]],   # M5: B11
        [[1, 1], [0, 0]],   # M6: B00+B01
        [[0, 0], [1, 1]]    # M7: B10+B11
    ]
    
    # Output combination coefficients for C blocks.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    coeff_C = [
        [1, 0, 0, 1],   # M1, M2, M3, M4, M5, M6, M7 coefficients for C00?
        # Wait, I need a 4x7 matrix coeff_C where coeff_C[c_block, r]
        # C00: M1(1) + M2(0) + M3(0) + M4(1) + M5(-1) + M6(0) + M7(1)
        # C01: M1(0) + M2(0) + M3(1) + M4(0) + M5(1) + M6(0) + M7(0)
        # C10: M1(0) + M2(1) + M3(0) + M4(1) + M5(0) + M6(0) + M7(0)
        # C11: M1(1) + M2(-1) + M3(1) + M4(0) + M5(0) + M6(1) + M7(0)
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Now construct U4, V4, W4.
    # For each r (0..6) and s (0..6):
    #   The term corresponds to the s-th product in the 2x2 decomposition of X_r * Y_r.
    #   U2[s] gives the coefficients for X_r's local indices.
    #   V2[s] gives the coefficients for Y_r's local indices.
    #   W2[s] gives the coefficients for the result Z_r's local indices.
    
    col = 0
    for r in range(7):
        for s in range(7):
            # U part
            # U2[s, i*2+j] is the coefficient for X_r[i,j]
            # X_r[i,j] = sum_{a,b} coeff_A[r][a][b] * A_{a*2+i, b*2+j}
            # So U4[global_a, col] += coeff_A[r][a][b] * U2[s, i*2+j]
            for i in range(2):
                for j in range(2):
                    u_val = U2[s, i*2+j]
                    if u_val != 0:
                        for a in range(2):
                            for b in range(2):
                                c_a = coeff_A[r][a][b]
                                if c_a != 0:
                                    global_i = a*2 + i
                                    global_j = b*2 + j
                                    U4[global_i*4 + global_j, col] += c_a * u_val
            
            # V part
            # V2[s, i*2+j] is the coefficient for Y_r[i,j]
            # Y_r[i,j] = sum_{a,b} coeff_B[r][a][b] * B_{a*2+i, b*2+j}
            for i in range(2):
                for j in range(2):
                    v_val = V2[s, i*2+j]
                    if v_val != 0:
                        for a in range(2):
                            for b in range(2):
                                c_b = coeff_B[r][a][b]
                                if c_b != 0:
                                    global_i = a*2 + i
                                    global_j = b*2 + j
                                    V4[global_i*4 + global_j, col] += c_b * v_val
            
            # W part
            # W2[s, k*2+i] is the coefficient for Z_r[k,i] (result of X_r * Y_r)
            # Z_r contributes to C blocks.
            # C_block_c = sum_r coeff_C[c, r] * Z_r
            # So W4[global_k, col] += coeff_C[c, r] * W2[s, k*2+i]
            # Global index for W is k*4+i.
            # Z_r is 2x2. Local indices (k, i) in 0..1.
            # This Z_r is added to C block c.
            # C block c has global rows c*2..c*2+1 and cols k_global.
            # Wait, Z_r[k, i] is the element at row k, col i of the 2x2 result.
            # This result is added to C_block_c.
            # C_block_c is at global rows c*2..c*2+1?
            # No, C is 4x4.
            # C00 is rows 0-1, cols 0-1.
            # C01 is rows 0-1, cols 2-3.
            # C10 is rows 2-3, cols 0-1.
            # C11 is rows 2-3, cols 2-3.
            
            # Z_r[k, i] corresponds to C_block_c[k, i].
            # Global row in C: if c=0 or 1, row = k. If c=2 or 3, row = 2+k.
            # Global col in C: if c=0 or 2, col = i. If c=1 or 3, col = 2+i.
            
            # Wait, the W index in the problem is k*n + i.
            # Here n=4. So W index is col*4 + row.
            # My Z_r[k, i] has row k, col i?
            # Standard matrix mult: (XY)_{row, col}.
            # In my 2x2 decomposition, W2[s, k*2+i] corresponds to output element (k, i)?
            # Let's check the 2x2 W definition.
            # W[k*2+i, c].
            # In _strassen_2x2, W[0,0]=1 (C00), W[3,0]=1 (C11).
            # C00 is row 0, col 0. Index 0*2+0=0.
            # C11 is row 1, col 1. Index 1*2+1=3.
            # So W index is col*2 + row.
            # So W2[s, k*2+i] corresponds to output element (row=i, col=k).
            
            # So Z_r[k, i] is row i, col k?
            # Let's re-verify Strassen 2x2.
            # C00 = M1...
            # C00 is row 0, col 0.
            # W[0, 0] = 1. Index 0.
            # C11 is row 1, col 1.
            # W[3, 0] = 1. Index 3 = 1*2+1.
            # So index = col*2 + row.
            
            # So W2[s, idx] corresponds to output (row=idx%2, col=idx//2).
            
            # For 4x4:
            # Z_r is 2x2. Its elements are (row_local, col_local).
            # Z_r is added to C_block_c.
            # C_block_c is at global (row_global, col_global).
            # If c=0 (C00): rows 0-1, cols 0-1.
            # If c=1 (C01): rows 0-1, cols 2-3.
            # If c=2 (C10): rows 2-3, cols 0-1.
            # If c=3 (C11): rows 2-3, cols 2-3.
            
            # For a local element (row_l, col_l) in Z_r:
            # global_row = (c//2)*2 + row_l
            # global_col = (c%2)*2 + col_l
            # W index = global_col*4 + global_row.
            
            for idx in range(4):
                w_val = W2[s, idx]
                if w_val != 0:
                    row_l = idx % 2
                    col_l = idx // 2
                    for c in range(4):
                        c_coeff = coeff_C[c, r]
                        if c_coeff != 0:
                            global_row = (c // 2) * 2 + row_l
                            global_col = (c % 2) * 2 + col_l
                            W4[global_col*4 + global_row, col] += c_coeff * w_val
            
            col += 1
            
    return U4, V4, W4

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    U2, V2, W2 = _strassen_2x2()
    U4, V4, W4 = _compose_4x4_from_2x2(U2, V2, W2)
    
    # Now we have a 4x4x4 decomposition of rank 49.
    # We need 4x4x8.
    # This is two independent 4x4x4 products.
    # B is 4x8. Split into B1 (cols 0-3) and B2 (cols 4-7).
    # C is 4x8. Split into C1 (cols 0-3) and C2 (cols 4-7).
    
    rank_total = 98
    U_total = np.zeros((16, rank_total))
    V_total = np.zeros((32, rank_total))
    V_total.shape = (32, rank_total) # m*p = 4*8 = 32
    W_total = np.zeros((32, rank_total)) # p*n = 8*4 = 32
    
    # Copy first 4x4 block
    U_total[:, 0:49] = U4
    # V4 is 16x49 (m*p for 4x4 is 4*4=16).
    # V_total is 32x98.
    # V4 corresponds to B1 (cols 0-3).
    # V_total rows 0-15 correspond to B1.
    V_total[0:16, 0:49] = V4
    # W4 is 16x49 (p*n for 4x4 is 4*4=16).
    # W_total is 32x98.
    # W4 corresponds to C1 (cols 0-3).
    # W_total rows 0-15 correspond to C1.
    W_total[0:16, 0:49] = W4
    
    # Copy second 4x4 block
    U_total[:, 49:98] = U4
    # V4 corresponds to B2 (cols 4-7).
    # V_total rows 16-31 correspond to B2.
    V_total[16:32, 49:98] = V4
    # W4 corresponds to C2 (cols 4-7).
    # W_total rows 16-31 correspond to C2.
    W_total[16:32, 49:98] = W4
    
    # Convert to lists of lists as required by interface
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
