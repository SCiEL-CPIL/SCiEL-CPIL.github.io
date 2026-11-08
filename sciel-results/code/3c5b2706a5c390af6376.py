import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Define the rank-7 decomposition for 2x2x2 matrix multiplication
    # U, V, W are 4x7 matrices (flattened 2x2)
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    U_2 = np.zeros((4, 7), dtype=int)
    V_2 = np.zeros((4, 7), dtype=int)
    W_2 = np.zeros((4, 7), dtype=int)
    
    # r=0: M1 = (a+d)(e+h)
    # X1 = a+d, Y1 = e+h
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    # C00 += M1, C11 += M1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # r=1: M2 = (c+d)e
    # X2 = c+d, Y2 = e
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    # C10 += M2, C11 -= M2
    W_2[1, 1] = 1; W_2[3, 1] = -1
    
    # r=2: M3 = a(f-h)
    # X3 = a, Y3 = f-h
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    # C01 += M3, C11 += M3
    W_2[2, 2] = 1; W_2[3, 2] = 1
    
    # r=3: M4 = d(g-c)
    # X4 = d, Y4 = g-c
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    # C00 += M4, C10 += M4
    W_2[0, 3] = 1; W_2[1, 3] = 1
    
    # r=4: M5 = (a+b)h
    # X5 = a+b, Y5 = h
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    # C00 -= M5, C01 += M5
    W_2[0, 4] = -1; W_2[2, 4] = 1
    
    # r=5: M6 = (c-a)(e+f)
    # X6 = c-a, Y6 = e+f
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    # C11 += M6
    W_2[3, 5] = 1
    
    # r=6: M7 = (b-d)(g+h)
    # X7 = b-d, Y7 = g+h
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    # C00 += M7
    W_2[0, 6] = 1
    
    # Now build 4x4x4 decomposition using Strassen recursion
    # Rank 49
    U_4 = np.zeros((16, 49), dtype=int)
    V_4 = np.zeros((16, 49), dtype=int)
    W_4 = np.zeros((16, 49), dtype=int)
    
    # Outer Strassen steps for 4x4x4
    # s=0: X1 = A00 + A11, Y1 = B00 + B11
    # s=1: X2 = A10 + A11, Y2 = B00
    # s=2: X3 = A00, Y3 = B01 - B11
    # s=3: X4 = A11, Y4 = B10 - B00
    # s=4: X5 = A00 + A01, Y5 = B11
    # s=5: X6 = A10 - A00, Y6 = B00 + B01
    # s=6: X7 = A01 - A11, Y7 = B10 + B11
    
    # For each outer step s, we have an inner 2x2x2 multiplication
    # The U, V, W for the inner multiplication are U_2, V_2, W_2
    # But we need to map the indices correctly.
    
    # The input A is 4x4. Blocks A00, A01, A10, A11 are 2x2.
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # For a given outer step s, X_s is a 4x4 matrix.
    # X_s is formed by linear combinations of A's blocks.
    # The inner multiplication treats X_s as a 2x2 block matrix of 2x2 blocks.
    # Let X_s = [[X00, X01], [X10, X11]] where Xij are 2x2.
    # The inner Strassen step t uses specific entries of X00, X01, X10, X11.
    # U_2[i*2+j, t] gives the coefficient for the (i,j)-th entry of the 2x2 block.
    # So we need to map the 2x2 block indices to the 4x4 matrix indices.
    
    # Let's define the block mapping for X_s
    # X_s_block[bi, bj] is a 2x2 matrix.
    # We need to know which A-block contributes to X_s_block[bi, bj] and with what coefficient.
    
    # A-blocks:
    # A00: (0,0) to (1,1)
    # A01: (0,2) to (1,3)
    # A10: (2,0) to (3,1)
    # A11: (2,2) to (3,3)
    
    # Outer step definitions (coefficients for A-blocks):
    # s=0: X1 = A00 + A11
    #   X100 = A00, X111 = A11, others 0
    # s=1: X2 = A10 + A11
    #   X210 = A10, X211 = A11
    # s=2: X3 = A00
    #   X300 = A00
    # s=3: X4 = A11
    #   X411 = A11
    # s=4: X5 = A00 + A01
    #   X500 = A00, X501 = A01
    # s=5: X6 = A10 - A00
    #   X610 = A10, X600 = -A00
    # s=6: X7 = A01 - A11
    #   X701 = A01, X711 = -A11
    
    # Similarly for Y_s (B-blocks)
    
    # For each s, we define a 2x2x2x2 array coeff_A[s, bi, bj, ai, aj]
    # where coeff_A[s, bi, bj, ai, aj] is the coefficient of A_block[ai, aj] in X_s_block[bi, bj].
    
    coeff_A = np.zeros((7, 2, 2, 2, 2), dtype=int)
    coeff_B = np.zeros((7, 2, 2, 2, 2), dtype=int)
    
    # s=0
    coeff_A[0, 0, 0, 0, 0] = 1 # X00 = A00
    coeff_A[0, 1, 1, 1, 1] = 1 # X11 = A11
    coeff_B[0, 0, 0, 0, 0] = 1 # Y00 = B00
    coeff_B[0, 1, 1, 1, 1] = 1 # Y11 = B11
    
    # s=1
    coeff_A[1, 1, 0, 1, 0] = 1 # X10 = A10
    coeff_A[1, 1, 1, 1, 1] = 1 # X11 = A11
    coeff_B[1, 0, 0, 0, 0] = 1 # Y00 = B00
    
    # s=2
    coeff_A[2, 0, 0, 0, 0] = 1 # X00 = A00
    coeff_B[2, 0, 1, 0, 1] = 1 # Y01 = B01
    coeff_B[2, 1, 1, 1, 1] = -1 # Y11 = -B11
    
    # s=3
    coeff_A[3, 1, 1, 1, 1] = 1 # X11 = A11
    coeff_B[3, 1, 0, 1, 0] = 1 # Y10 = B10
    coeff_B[3, 0, 0, 0, 0] = -1 # Y00 = -B00
    
    # s=4
    coeff_A[4, 0, 0, 0, 0] = 1 # X00 = A00
    coeff_A[4, 0, 1, 0, 1] = 1 # X01 = A01
    coeff_B[4, 1, 1, 1, 1] = 1 # Y11 = B11
    
    # s=5
    coeff_A[5, 1, 0, 1, 0] = 1 # X10 = A10
    coeff_A[5, 0, 0, 0, 0] = -1 # X00 = -A00
    coeff_B[5, 0, 0, 0, 0] = 1 # Y00 = B00
    coeff_B[5, 0, 1, 0, 1] = 1 # Y01 = B01
    
    # s=6
    coeff_A[6, 0, 1, 0, 1] = 1 # X01 = A01
    coeff_A[6, 1, 1, 1, 1] = -1 # X11 = -A11
    coeff_B[6, 1, 0, 1, 0] = 1 # Y10 = B10
    coeff_B[6, 1, 1, 1, 1] = 1 # Y11 = B11
    
    # Now fill U_4, V_4, W_4
    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            # U_4 entries
            # U_2[i*2+j, t] is the coefficient for the (i,j)-th entry of the 2x2 block.
            # The 2x2 block is X_s_block[bi, bj].
            # X_s_block[bi, bj] is a linear combination of A_blocks.
            # A_block[ai, aj] corresponds to rows 2*ai to 2*ai+1, cols 2*aj to 2*aj+1.
            for bi in range(2):
                for bj in range(2):
                    for ai in range(2):
                        for aj in range(2):
                            c = coeff_A[s, bi, bj, ai, aj]
                            if c != 0:
                                # Add c * U_2 entries for the 2x2 block
                                for ii in range(2):
                                    for jj in range(2):
                                        u_val = U_2[ii*2+jj, t]
                                        if u_val != 0:
                                            # Row in U_4: 2*ai + ii, Col in U_4: 2*aj + jj
                                            row = 2*ai + ii
                                            col = 2*aj + jj
                                            U_4[row, r] += c * u_val
                                            
            # V_4 entries
            for bi in range(2):
                for bj in range(2):
                    for ai in range(2):
                        for aj in range(2):
                            c = coeff_B[s, bi, bj, ai, aj]
                            if c != 0:
                                for ii in range(2):
                                    for jj in range(2):
                                        v_val = V_2[ii*2+jj, t]
                                        if v_val != 0:
                                            row = 2*ai + ii
                                            col = 2*aj + jj
                                            V_4[row, r] += c * v_val
                                            
            # W_4 entries
            # W_2[k*2+i, t] is the coefficient for C_block[i,k] in the 2x2 result.
            # The 2x2 result block is P_s_block[i,k].
            # P_s_block[i,k] is the i,k-th block of the 4x4 product P_s.
            # So it contributes to W_4 at row 2*k + ii, col 2*i + jj?
            # Wait, W_4 index is k*n + i. n=4.
            # The block P_s_block[i,k] is at rows 2*k to 2*k+1, cols 2*i to 2*i+1.
            # So the entry (ii, jj) of P_s_block[i,k] is at row 2*k+ii, col 2*i+jj in the 4x4 matrix.
            # The index in W_4 is (2*k+ii) * 4 + (2*i+jj).
            for ii in range(2):
                for jj in range(2):
                    w_val = W_2[jj*2+ii, t] # Note: W_2 index is k*2+i. Here i=ii, k=jj?
                    # In W_2, the index is k*2+i.
                    # The block is P_s_block[i,k].
                    # So the row in the block is i (which is ii), col is k (which is jj).
                    # So we need W_2[jj*2+ii, t].
                    if w_val != 0:
                        # The block P_s_block[i,k] is at rows 2*k to 2*k+1, cols 2*i to 2*i+1.
                        # Wait, C = AB. C[i,k] is the i-th row, k-th col.
                        # In the 4x4 matrix, the block (i,k) is at rows 2*i to 2*i+1, cols 2*k to 2*k+1.
                        # So the entry (ii, jj) is at row 2*i+ii, col 2*k+jj.
                        # The index in W_4 is row * 4 + col = (2*i+ii)*4 + (2*k+jj).
                        # Here i=ii, k=jj.
                        row = 2*ii + ii # No, i is the block row index.
                        # Let's rename block indices to br, bc.
                        # W_2 index is bc*2 + br.
                        # So w_val = W_2[jj*2+ii, t] is for block (ii, jj).
                        # Block (ii, jj) is at rows 2*ii to 2*ii+1, cols 2*jj to 2*jj+1.
                        # Entry (ii, jj) inside the block is at row 2*ii + 0, col 2*jj + 0?
                        # No, the block is 2x2. The entries are (0,0), (0,1), (1,0), (1,1).
                        # Let's iterate over the entries of the 2x2 block.
                        pass
                        
    # Redo W_4 filling
    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            for br in range(2): # Block row
                for bc in range(2): # Block col
                    w_val = W_2[bc*2+br, t]
                    if w_val != 0:
                        # The block is at rows 2*br to 2*br+1, cols 2*bc to 2*bc+1.
                        for ii in range(2):
                            for jj in range(2):
                                row = 2*br + ii
                                col = 2*bc + jj
                                W_4[row*4+col, r] += w_val
                                
    # Now we have U_4, V_4, W_4 for 4x4x4.
    # We need 4x4x8.
    # We split B into B1, B2 (each 4x4).
    # C into C1, C2 (each 4x4).
    # A is the same.
    # We use the 4x4x4 decomposition twice.
    
    rank = 98
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((32, rank), dtype=int)
    W = np.zeros((32, rank), dtype=int)
    
    # First half: C1 = A B1
    U[:, 0:49] = U_4
    V[0:16, 0:49] = V_4
    W[0:16, 0:49] = W_4
    
    # Second half: C2 = A B2
    U[:, 49:98] = U_4
    V[16:32, 49:98] = V_4
    W[16:32, 49:98] = W_4
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
