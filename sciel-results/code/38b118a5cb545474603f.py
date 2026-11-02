import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Construct rank 49 decomposition for 4x4x4
    # Indices for 4x4: i, j, k in 0..3
    # U4 shape (16, 49), V4 shape (16, 49), W4 shape (16, 49)
    # Note: V4 and W4 indices are j*4+k and k*4+i for the 4x4 case
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Strassen 4x4 level: 7 products M_s
    # L_A_s: linear forms of A blocks
    # L_B_s: linear forms of B blocks
    # c_s: coefficients for C blocks
    
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11
    
    # Define the 7 linear forms for A and B
    # We represent a linear form as a 2x2 matrix of coefficients for the 4 blocks?
    # No, easier to just define the 2x2 matrix X_s directly in terms of A's entries.
    # But we need to extract the linear form for each entry of X_s.
    
    # Let's define the 7 products P_t for 2x2 Strassen
    # P1 = (X00 + X11)(Y00 + Y11)
    # P2 = (X10 + X11)Y00
    # P3 = X00(Y01 - Y11)
    # P4 = X11(Y10 - Y00)
    # P5 = (X00 + X01)Y11
    # P6 = (X10 - X00)(Y00 + Y01)
    # P7 = (X01 - X11)(Y10 + Y11)
    
    # We need to map X_s entries to A entries.
    # X_s is a 2x2 matrix.
    # For s=1: X1 = A00 + A11. So X1_00 = A00_00 + A11_00, etc.
    
    # Let's create a helper to get the linear form of A for a given X_s entry.
    # A is 4x4. Blocks are 2x2.
    # A[i,j] where i,j in 0..3.
    # Block A_{kl} is A[2k:2k+2, 2l:2l+2].
    
    # We will iterate s in 0..6
    for s in range(7):
        # Determine X_s and Y_s definitions
        # X_s is sum/diff of A blocks
        # Y_s is sum/diff of B blocks
        
        # Coefficients for A blocks in X_s
        # X_s is 2x2. Each entry is a sum of A block entries.
        # We represent X_s definition as a 2x2x2x2 array? No.
        # Let's just define the 2x2 matrix of block indices and signs.
        
        # s=0 (M1): X = A00 + A11
        # s=1 (M2): X = A10 + A11
        # s=2 (M3): X = A00
        # s=3 (M4): X = A11
        # s=4 (M5): X = A00 + A01
        # s=5 (M6): X = A10 - A00
        # s=6 (M7): X = A01 - A11
        
        # Similarly for Y_s with B blocks.
        
        # Let's define a function to get the linear form for X_s[i,j]
        # It returns a 4x4 matrix L such that X_s[i,j] = sum L[a,b] A[a,b]
        
        def get_A_form(s, xi, xj):
            # Returns 4x4 matrix of coefficients
            L = np.zeros((4,4))
            if s == 0: # A00 + A11
                if xi == 0 and xj == 0:
                    L[0:2, 0:2] += 1
                    L[2:4, 2:4] += 1
                elif xi == 0 and xj == 1:
                    L[0:2, 2:4] += 1
                    L[2:4, 0:2] += 1 # Wait, A11 is bottom-right. A00 is top-left.
                    # A00 is [0:2, 0:2]. A11 is [2:4, 2:4].
                    # X_s is 2x2 matrix of blocks? No, X_s is a 2x2 matrix.
                    # In Strassen, X_s is a matrix.
                    # For M1, X1 = A00 + A11. This is a 4x4 matrix?
                    # No, in the 4x4 Strassen, the "blocks" are 2x2.
                    # So A00 is a 2x2 matrix.
                    # X1 is a 2x2 matrix?
                    # If A is 4x4, partitioned into 2x2 blocks.
                    # A00 is top-left 2x2.
                    # X1 = A00 + A11. This is a 2x2 matrix (sum of two 2x2 matrices).
                    # So X1[i,j] is a scalar.
                    # My previous thought was wrong. X_s is a 2x2 matrix of scalars?
                    # No, A00 is a 2x2 matrix. Sum of two 2x2 matrices is a 2x2 matrix.
                    # So X_s is a 2x2 matrix.
                    # Its entries are linear forms of the 16 entries of A.
                    # So L is 4x4.
                    pass
            
            # Let's re-evaluate.
            # A is 4x4.
            # Blocks: A00 (0:2, 0:2), A01 (0:2, 2:4), A10 (2:4, 0:2), A11 (2:4, 2:4).
            # X_s is a 2x2 matrix.
            # For s=0, X0 = A00 + A11.
            # X0[0,0] = A00[0,0] + A11[0,0] = A[0,0] + A[2,2].
            # X0[0,1] = A00[0,1] + A11[0,1] = A[0,1] + A[2,3].
            # X0[1,0] = A00[1,0] + A11[1,0] = A[1,0] + A[3,2].
            # X0[1,1] = A00[1,1] + A11[1,1] = A[1,1] + A[3,3].
            
            # So for each entry (xi, xj) of X_s, we can determine the 4x4 coefficient matrix.
            
            # Define block offsets
            # Block (b_i, b_j) is at [2*b_i : 2*b_i+2, 2*b_j : 2*b_j+2]
            
            # s=0: A00 + A11
            # s=1: A10 + A11
            # s=2: A00
            # s=3: A11
            # s=4: A00 + A01
            # s=5: A10 - A00
            # s=6: A01 - A11
            
            # For each s, define the blocks involved and signs.
            blocks = {
                0: [(0,0,1), (1,1,1)],
                1: [(1,0,1), (1,1,1)],
                2: [(0,0,1)],
                3: [(1,1,1)],
                4: [(0,0,1), (0,1,1)],
                5: [(1,0,1), (0,0,-1)],
                6: [(0,1,1), (1,1,-1)]
            }
            
            L = np.zeros((4,4))
            for (bi, bj, sign) in blocks[s]:
                L[2*bi:2*bi+2, 2*bj:2*bj+2] += sign
            return L
        
        # Similarly for B
        def get_B_form(s, yi, yj):
            # Y_s blocks are the same structure
            blocks = {
                0: [(0,0,1), (1,1,1)],
                1: [(0,0,1)], # Y1 = B00
                2: [(0,1,1), (1,1,-1)], # Y2 = B01 - B11
                3: [(1,0,1), (0,0,-1)], # Y3 = B10 - B00
                4: [(1,1,1)], # Y4 = B11
                5: [(0,0,1), (0,1,1)], # Y5 = B00 + B01
                6: [(1,0,1), (1,1,1)] # Y6 = B10 + B11
            }
            # Wait, check Strassen definitions again.
            # M1 = (A00+A11)(B00+B11) -> Y0 = B00+B11
            # M2 = (A10+A11)B00 -> Y1 = B00
            # M3 = A00(B01-B11) -> Y2 = B01-B11
            # M4 = A11(B10-B00) -> Y3 = B10-B00
            # M5 = (A00+A01)B11 -> Y4 = B11
            # M6 = (A10-A00)(B00+B01) -> Y5 = B00+B01
            # M7 = (A01-A11)(B10+B11) -> Y6 = B10+B11
            
            # Corrected blocks for B:
            blocks = {
                0: [(0,0,1), (1,1,1)],
                1: [(0,0,1)],
                2: [(0,1,1), (1,1,-1)],
                3: [(1,0,1), (0,0,-1)],
                4: [(1,1,1)],
                5: [(0,0,1), (0,1,1)],
                6: [(1,0,1), (1,1,1)]
            }
            
            L = np.zeros((4,4))
            for (bi, bj, sign) in blocks[s]:
                L[2*bi:2*bi+2, 2*bj:2*bj+2] += sign
            return L

        # Now, for each s, we have X_s and Y_s.
        # We decompose X_s Y_s using 2x2 Strassen (7 terms).
        # The 7 terms P_t are defined by linear forms of X_s and Y_s entries.
        
        # P1 = (X00+X11)(Y00+Y11)
        # P2 = (X10+X11)Y00
        # P3 = X00(Y01-Y11)
        # P4 = X11(Y10-Y00)
        # P5 = (X00+X01)Y11
        # P6 = (X10-X00)(Y00+Y01)
        # P7 = (X01-X11)(Y10+Y11)
        
        # For each t in 0..6, we determine the linear form of A and B.
        # L_A_t = sum of L_A(X_s[i,j]) with coefficients from P_t definition.
        
        for t in range(7):
            # Determine coefficients for X_s entries
            # P_t = (sum c_x[i,j] X_s[i,j]) * (sum c_y[i,j] Y_s[i,j])
            
            # Define the structure of P_t
            # t=0: X: (0,0,1), (1,1,1); Y: (0,0,1), (1,1,1)
            # t=1: X: (1,0,1), (1,1,1); Y: (0,0,1)
            # t=2: X: (0,0,1); Y: (0,1,1), (1,1,-1)
            # t=3: X: (1,1,1); Y: (1,0,1), (0,0,-1)
            # t=4: X: (0,0,1), (0,1,1); Y: (1,1,1)
            # t=5: X: (1,0,1), (0,0,-1); Y: (0,0,1), (0,1,1)
            # t=6: X: (0,1,1), (1,1,-1); Y: (1,0,1), (1,1,1)
            
            x_struct = [
                [(0,0,1), (1,1,1)],
                [(1,0,1), (1,1,1)],
                [(0,0,1)],
                [(1,1,1)],
                [(0,0,1), (0,1,1)],
                [(1,0,1), (0,0,-1)],
                [(0,1,1), (1,1,-1)]
            ][t]
            
            y_struct = [
                [(0,0,1), (1,1,1)],
                [(0,0,1)],
                [(0,1,1), (1,1,-1)],
                [(1,0,1), (0,0,-1)],
                [(1,1,1)],
                [(0,0,1), (0,1,1)],
                [(1,0,1), (1,1,1)]
            ][t]
            
            # Compute L_A for this term
            L_A = np.zeros((4,4))
            for (xi, xj, sign) in x_struct:
                L_A += sign * get_A_form(s, xi, xj)
                
            L_B = np.zeros((4,4))
            for (yi, yj, sign) in y_struct:
                L_B += sign * get_B_form(s, yi, yj)
            
            # Now we have L_A and L_B (4x4 matrices).
            # These are the linear forms for the U and V parts of this term.
            # Term index c = s * 7 + t
            c = s * 7 + t
            
            # Fill U4
            # U4[i*4+j, c] = L_A[i,j]
            for i in range(4):
                for j in range(4):
                    U4[i*4+j, c] = L_A[i,j]
                    
            # Fill V4
            # V4[j*4+k, c] = L_B[j,k]
            for j in range(4):
                for k in range(4):
                    V4[j*4+k, c] = L_B[j,k]
            
            # Fill W4
            # W4[k*4+i, c] = weight
            # Weight is c_s,kl * w_t,alpha,beta
            # c_s,kl is the coefficient of M_s in C_kl
            # w_t,alpha,beta is the coefficient of P_t in M_s,alpha,beta
            
            # C reconstruction:
            # C00 = M1 + M4 - M5 + M7 -> s=0:1, s=3:1, s=4:-1, s=6:1
            # C01 = M3 + M5 -> s=2:1, s=4:1
            # C10 = M2 + M4 -> s=1:1, s=3:1
            # C11 = M1 - M2 + M3 + M6 -> s=0:1, s=1:-1, s=2:1, s=5:1
            
            c_coeffs = {
                (0,0): {0:1, 3:1, 4:-1, 6:1},
                (0,1): {2:1, 4:1},
                (1,0): {1:1, 3:1},
                (1,1): {0:1, 1:-1, 2:1, 5:1}
            }
            
            # M_s reconstruction from P_t:
            # M00 = P1 + P4 - P5 + P7 -> t=0:1, t=3:1, t=4:-1, t=6:1
            # M01 = P3 + P5 -> t=2:1, t=4:1
            # M10 = P2 + P4 -> t=1:1, t=3:1
            # M11 = P1 - P2 + P3 + P6 -> t=0:1, t=1:-1, t=2:1, t=5:1
            
            m_coeffs = {
                (0,0): {0:1, 3:1, 4:-1, 6:1},
                (0,1): {2:1, 4:1},
                (1,0): {1:1, 3:1},
                (1,1): {0:1, 1:-1, 2:1, 5:1}
            }
            
            for kl_i in range(2):
                for kl_j in range(2):
                    # C block index (kl_i, kl_j)
                    c_coeff = c_coeffs[(kl_i, kl_j)].get(s, 0)
                    if c_coeff == 0: continue
                    
                    for alpha in range(2):
                        for beta in range(2):
                            # M_s entry (alpha, beta)
                            m_coeff = m_coeffs[(alpha, beta)].get(t, 0)
                            if m_coeff == 0: continue
                            
                            weight = c_coeff * m_coeff
                            
                            # Global indices for W
                            # C index: I = 2*kl_i + alpha, K = 2*kl_j + beta
                            I = 2*kl_i + alpha
                            K = 2*kl_j + beta
                            
                            # W4 index: K*4 + I
                            W4[K*4 + I, c] = weight

    # 2. Assemble full U, V, W for 4x4x8
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # First 49 columns
    U[:, 0:49] = U4
    # V: k in 0..3
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j*4 + k, 0:49]
    # W: k in 0..3
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W4[k*4 + i, 0:49]
            
    # Last 49 columns
    U[:, 49:98] = U4
    # V: k in 4..7
    for j in range(4):
        for k in range(4, 8):
            V[j*8 + k, 49:98] = V4[j*4 + (k-4), 0:49]
    # W: k in 4..7
    for k in range(4, 8):
        for i in range(4):
            W[k*4 + i, 49:98] = W4[(k-4)*4 + i, 0:49]
            
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
