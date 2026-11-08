import numpy as np
import time

def construct():
    n, m, p = 4, 4, 5
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # We will build a decomposition using Strassen's algorithm for the 4x4x4 part
    # and schoolbook for the 5th column.
    # Total rank = 49 (Strassen 4x4x4) + 16 (Schoolbook for 5th col) = 65
    
    rank_total = 65
    U = np.zeros((n * m, rank_total))  # 16 x 65
    V = np.zeros((m * p, rank_total))  # 20 x 65
    W = np.zeros((p * n, rank_total))  # 20 x 65

    # --- Part 1: Strassen 4x4x4 (Rank 49) ---
    # We decompose the 4x4x4 multiplication tensor into 49 rank-1 terms.
    # This is done by applying Strassen 2x2x2 (rank 7) at two levels.
    
    # First, define the Strassen 2x2x2 factors (U2, V2, W2) of shape (4, 7).
    # Indices: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen formulas for 2x2:
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = (A10+A11)B00       -> C10 += M2, C11 -= M2
    # M3 = A00(B01-B11)       -> C01 += M3, C11 += M3
    # M4 = A11(B10-B00)       -> C00 += M4, C10 += M4
    # M5 = (A00+A01)B11       -> C00 -= M5, C01 += M5
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    
    # Term 1: M1
    U2[0, 0] = 1; U2[3, 0] = 1  # A00, A11
    V2[0, 0] = 1; V2[3, 0] = 1  # B00, B11
    W2[0, 0] = 1; W2[3, 0] = 1  # C00, C11
    
    # Term 2: M2
    U2[2, 1] = 1; U2[3, 1] = 1  # A10, A11
    V2[0, 1] = 1                # B00
    W2[2, 1] = 1; W2[3, 1] = -1 # C10, -C11
    
    # Term 3: M3
    U2[0, 2] = 1                # A00
    V2[1, 2] = 1; V2[3, 2] = -1 # B01, -B11
    W2[1, 2] = 1; W2[3, 2] = 1  # C01, C11
    
    # Term 4: M4
    U2[3, 3] = 1                # A11
    V2[2, 3] = 1; V2[0, 3] = -1 # B10, -B00
    W2[0, 3] = 1; W2[2, 3] = 1  # C00, C10
    
    # Term 5: M5
    U2[0, 4] = 1; U2[1, 4] = 1  # A00, A01
    V2[3, 4] = 1                # B11
    W2[0, 4] = -1; W2[1, 4] = 1 # -C00, C01
    
    # Term 6: M6
    U2[2, 5] = 1; U2[0, 5] = -1 # A10, -A00
    V2[0, 5] = 1; V2[1, 5] = 1  # B00, B01
    W2[3, 5] = 1                # C11
    
    # Term 7: M7
    U2[1, 6] = 1; U2[3, 6] = -1 # A01, -A11
    V2[2, 6] = 1; V2[3, 6] = 1  # B10, B11
    W2[0, 6] = 1                # C00

    # Now, construct the 49 terms for 4x4x4.
    # We view A and B as 2x2 block matrices with 2x2 blocks.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # The top-level Strassen terms are:
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    # Each S_s is a product of two 4x4 matrices (formed from blocks).
    # Let L_s and R_s be the 4x4 matrices.
    # We decompose L_s * R_s using the 2x2 Strassen structure on the blocks of L_s and R_s.
    # This generates 7 inner terms for each outer term s.
    # Total 7 * 7 = 49 terms.
    
    col = 0
    for s in range(7):
        # Define L_s and R_s as 4x4 matrices of coefficients (0, 1, -1)
        L_s = np.zeros((4, 4))
        R_s = np.zeros((4, 4))
        
        if s == 0:
            # L = A00+A11, R = B00+B11
            L_s[0:2, 0:2] = 1 # A00
            L_s[2:4, 2:4] = 1 # A11
            R_s[0:2, 0:2] = 1 # B00
            R_s[2:4, 2:4] = 1 # B11
        elif s == 1:
            # L = A10+A11, R = B00
            L_s[2:4, 0:2] = 1 # A10
            L_s[2:4, 2:4] = 1 # A11
            R_s[0:2, 0:2] = 1 # B00
        elif s == 2:
            # L = A00, R = B01-B11
            L_s[0:2, 0:2] = 1 # A00
            R_s[0:2, 2:4] = 1 # B01
            R_s[2:4, 2:4] = -1 # -B11
        elif s == 3:
            # L = A11, R = B10-B00
            L_s[2:4, 2:4] = 1 # A11
            R_s[2:4, 0:2] = 1 # B10
            R_s[0:2, 0:2] = -1 # -B00
        elif s == 4:
            # L = A00+A01, R = B11
            L_s[0:2, 0:2] = 1 # A00
            L_s[0:2, 2:4] = 1 # A01
            R_s[2:4, 2:4] = 1 # B11
        elif s == 5:
            # L = A10-A00, R = B00+B01
            L_s[2:4, 0:2] = 1 # A10
            L_s[0:2, 0:2] = -1 # -A00
            R_s[0:2, 0:2] = 1 # B00
            R_s[0:2, 2:4] = 1 # B01
        elif s == 6:
            # L = A01-A11, R = B10+B11
            L_s[0:2, 2:4] = 1 # A01
            L_s[2:4, 2:4] = -1 # -A11
            R_s[2:4, 0:2] = 1 # B10
            R_s[2:4, 2:4] = 1 # B11

        # Now decompose L_s * R_s.
        # L_s and R_s are 4x4 matrices.
        # We treat them as 2x2 block matrices with 2x2 blocks.
        # Apply Strassen 2x2x2 to these block matrices.
        
        # Extract blocks
        L00 = L_s[0:2, 0:2]
        L01 = L_s[0:2, 2:4]
        L10 = L_s[2:4, 0:2]
        L11 = L_s[2:4, 2:4]
        
        R00 = R_s[0:2, 0:2]
        R01 = R_s[0:2, 2:4]
        R10 = R_s[2:4, 0:2]
        R11 = R_s[2:4, 2:4]
        
        # Inner Strassen terms
        # t=0: (L00+L11)(R00+R11)
        # t=1: (L10+L11)R00
        # t=2: L00(R01-R11)
        # t=3: L11(R10-R00)
        # t=4: (L00+L01)R11
        # t=5: (L10-L00)(R00+R01)
        # t=6: (L01-L11)(R10+R11)
        
        for t in range(7):
            if time.time() > deadline:
                break
                
            # Form the 2x2 matrices X and Y for the inner product
            if t == 0:
                X = L00 + L11
                Y = R00 + R11
            elif t == 1:
                X = L10 + L11
                Y = R00
            elif t == 2:
                X = L00
                Y = R01 - R11
            elif t == 3:
                X = L11
                Y = R10 - R00
            elif t == 4:
                X = L00 + L01
                Y = R11
            elif t == 5:
                X = L10 - L00
                Y = R00 + R01
            elif t == 6:
                X = L01 - L11
                Y = R10 + R11
            
            # Now we have a 2x2 matrix product X*Y.
            # We use U2, V2, W2 to decompose this into 7 rank-1 terms.
            # The contribution to the global U, V, W is:
            # U[i*4+j, col] += X[i,j] * U2[i*2+j, tp]
            # V[j*4+k, col] += Y[j,k] * V2[j*2+k, tp]
            # W[k*4+i, col] += W2[k*2+i, tp]
            
            for tp in range(7):
                if col >= rank_total or time.time() > deadline:
                    break
                    
                # Fill U
                for i in range(2):
                    for j in range(2):
                        if X[i, j] != 0:
                            U[i * 4 + j, col] += X[i, j] * U2[i * 2 + j, tp]
                
                # Fill V
                for j in range(2):
                    for k in range(2):
                        if Y[j, k] != 0:
                            V[j * 4 + k, col] += Y[j, k] * V2[j * 2 + k, tp]
                
                # Fill W
                for k in range(2):
                    for i in range(2):
                        W[k * 4 + i, col] += W2[k * 2 + i, tp]
                
                col += 1

    # --- Part 2: Schoolbook for the 5th column (Rank 16) ---
    # C[i,4] = sum_j A[i,j] B[j,4]
    # We need 16 terms for i=0..3, j=0..3.
    # U[i*4+j, col] = 1
    # V[j*5+4, col] = 1
    # W[4*4+i, col] = 1
    
    for i in range(4):
        for j in range(4):
            if col >= rank_total or time.time() > deadline:
                break
            U[i * 4 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
