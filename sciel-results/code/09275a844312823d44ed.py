import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition recursively.
    # The function build_decomp(n, m, p) returns U, V, W for M_{n,m,p}
    # and the rank.
    
    def build_decomp(n, m, p):
        # Base cases or small cases handled by specific logic
        # If n, m, p are small, use schoolbook
        if n == 1 and m == 1 and p == 1:
            U = np.ones((1, 1))
            V = np.ones((1, 1))
            W = np.ones((1, 1))
            return U, V, W, 1
        
        # Special case for 3x2x2
        if n == 3 and m == 2 and p == 2:
            return build_3x2x2()
            
        # Special case for 3x1x6 (or generally 3x1xp)
        if n == 3 and m == 1:
            # C = A B, A is 3x1, B is 1xp
            # C_ik = A_i1 B_1k
            # Rank is 3*p
            r = n * p
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            col = 0
            for i in range(n):
                for k in range(p):
                    U[i, col] = 1
                    V[k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
            return U, V, W, r

        # General recursive split
        # Split m into m1, m2
        # Split p into p1, p2? No, split based on m and p to balance.
        
        # Heuristic: split the middle dimension m
        if m >= 2:
            m1 = m // 2
            m2 = m - m1
            # A is n x m1, n x m2
            # B is m1 x p, m2 x p
            
            U1, V1, W1, r1 = build_decomp(n, m1, p)
            U2, V2, W2, r2 = build_decomp(n, m2, p)
            
            # Combine
            U = np.zeros((n * m, r1 + r2))
            V = np.zeros((m * p, r1 + r2))
            W = np.zeros((p * n, r1 + r2))
            
            # Copy U1, V1, W1
            U[:n * m1, :r1] = U1
            V[:m1 * p, :r1] = V1
            W[:, :r1] = W1
            
            # Copy U2, V2, W2 with offset
            U[n * m1:, r1:] = U2
            V[m1 * p:, r1:] = V2
            W[:, r1:] = W2
            
            return U, V, W, r1 + r2
            
        # If m == 1, handled above
        # If m < 2, error
        raise ValueError("Invalid dimensions")

    def build_3x2x2():
        # Strassen for top 2x2x2 (rank 7) + 4 extra for 3rd row
        # Top 2x2x2
        # A_top is 2x2, B is 2x2, C_top is 2x2
        # Strassen U, V, W for 2x2x2
        U_s = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            # ... need actual Strassen matrices
        ])
        # This is getting complex to hardcode. Let's just use schoolbook for 3x2x2?
        # Schoolbook rank is 12. Strassen+4 is 11.
        # Let's implement Strassen for 2x2x2 properly.
        
        # Strassen 2x2x2
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: A: 0,1,2,3 (row-major 2x2), B: 0,1,2,3, C: 0,1,2,3
        U_s = np.zeros((4, 7))
        V_s = np.zeros((4, 7))
        W_s = np.zeros((4, 7))
        
        # P1 = (A11+A22)(B11+B22)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[1, 0] = 1; W_s[2, 0] = 1; W_s[3, 0] = 1 # C11, C12, C21, C22 contributions?
        # Actually W maps to C.
        # C11 = P1 - P3 - P4 + P7
        # C12 = P5 + P7
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 - P5
        
        # Let's define the 7 terms
        # T0: (A00+A11)(B00+B11)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[1, 0] = 1; W_s[2, 0] = 1; W_s[3, 0] = 1 # This is wrong. W should be sparse.
        
        # Correct Strassen W:
        # C00 = T0 - T2 - T3 + T6
        # C01 = T4 + T6
        # C10 = T1 + T3
        # C11 = T0 - T1 + T2 - T4
        
        # T0: (A00+A11)(B00+B11)
        U_s[0, 0] = 1; U_s[3, 0] = 1
        V_s[0, 0] = 1; V_s[3, 0] = 1
        W_s[0, 0] = 1; W_s[1, 0] = 1; W_s[2, 0] = 1; W_s[3, 0] = 1 # No, W is 4x7.
        # W[k*n+i, c]. For 2x2, n=2.
        # C00 is W[0,0], C01 is W[1,0], C10 is W[2,0], C11 is W[3,0].
        # T0 contributes to all 4? No.
        # T0 = (A00+A11)(B00+B11).
        # This term is used in C00, C01, C10, C11 with coefficients.
        # So W column 0 should be [1, 1, 1, 1]?
        # Let's check:
        # C00 += 1 * T0
        # C01 += 1 * T0? No.
        # C01 = T4 + T6. T0 is not in C01.
        # So W[1, 0] should be 0.
        
        # Let's re-derive Strassen W.
        # T0: (A00+A11)(B00+B11)
        # T1: (A10+A11)B00
        # T2: A00(B10-B11)
        # T3: A11(B01-B00)
        # T4: (A00+A01)B11
        # T5: (A10-A00)(B00+B01)
        # T6: (A01-A11)(B10+B11)
        
        # C00 = T0 - T2 - T3 + T6
        # C01 = T4 + T6
        # C10 = T1 + T3
        # C11 = T0 - T1 + T2 - T4
        
        # W matrix (4x7):
        # Row 0 (C00): [1, 0, -1, -1, 0, 0, 1]
        # Row 1 (C01): [0, 0, 0, 0, 1, 0, 1]
        # Row 2 (C10): [0, 1, 0, 1, 0, 0, 0]
        # Row 3 (C11): [1, -1, 1, 0, -1, 0, 0]
        
        W_s[0, :] = [1, 0, -1, -1, 0, 0, 1]
        W_s[1, :] = [0, 0, 0, 0, 1, 0, 1]
        W_s[2, :] = [0, 1, 0, 1, 0, 0, 0]
        W_s[3, :] = [1, -1, 1, 0, -1, 0, 0]
        
        # U matrix (4x7):
        # Col 0: A00, A11 -> [1, 0, 0, 1]
        # Col 1: A10, A11 -> [0, 0, 1, 1]
        # Col 2: A00 -> [1, 0, 0, 0]
        # Col 3: A11 -> [0, 0, 0, 1]
        # Col 4: A00, A01 -> [1, 1, 0, 0]
        # Col 5: A10, -A00 -> [-1, 0, 1, 0]
        # Col 6: A01, -A11 -> [0, 1, 0, -1]
        
        U_s[0, :] = [1, 0, 1, 0, 1, -1, 0]
        U_s[1, :] = [0, 0, 0, 0, 1, 0, 1]
        U_s[2, :] = [0, 1, 0, 0, 0, 1, 0]
        U_s[3, :] = [1, 1, 0, 1, 0, 0, -1]
        
        # V matrix (4x7):
        # Col 0: B00, B11 -> [1, 0, 0, 1]
        # Col 1: B00 -> [1, 0, 0, 0]
        # Col 2: B10, -B11 -> [0, 1, 0, -1]
        # Col 3: B01, -B00 -> [-1, 0, 1, 0]
        # Col 4: B11 -> [0, 0, 0, 1]
        # Col 5: B00, B01 -> [1, 1, 0, 0]
        # Col 6: B10, B11 -> [0, 1, 0, 1]
        
        V_s[0, :] = [1, 1, 0, -1, 0, 1, 0]
        V_s[1, :] = [0, 0, 1, 0, 0, 1, 1]
        V_s[2, :] = [0, 0, 0, 1, 0, 0, 0]
        V_s[3, :] = [1, 0, -1, 0, 1, 0, 1]
        
        # Now add the 3rd row of A (index 4 in 3x2)
        # A30, A31
        # C30 = A30 B00 + A31 B10
        # C31 = A30 B01 + A31 B11
        # 4 extra terms
        U_e = np.zeros((2, 4))
        V_e = np.zeros((4, 4))
        W_e = np.zeros((4, 4))
        
        # T7: A30 B00
        U_e[0, 0] = 1
        V_e[0, 0] = 1
        W_e[4, 0] = 1 # C30 is index 4 in 3x2 output?
        # Output C is 3x2. Indices: 0,1 (row 0), 2,3 (row 1), 4,5 (row 2).
        # W is p*n x r = 2*3 x r = 6 x r.
        # W index for C30 is 0*3 + 2 = 2?
        # W[k*n+i]. k=0, i=2 -> 2.
        # C31 is k=1, i=2 -> 1*3+2 = 5.
        
        W_e = np.zeros((6, 4))
        U_e = np.zeros((6, 4)) # A is 3x2 -> 6 entries? No, A is 3x2.
        # U is n*m x r. n=3, m=2 -> 6 x r.
        # A30 is index 0*2 + 2 = 2? No, row-major.
        # A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3, A[2,0]=4, A[2,1]=5.
        # So A30 is index 4, A31 is index 5.
        
        U_e = np.zeros((6, 4))
        V_e = np.zeros((4, 4))
        W_e = np.zeros((6, 4))
        
        # T7: A30 B00 -> C30
        U_e[4, 0] = 1
        V_e[0, 0] = 1
        W_e[2, 0] = 1
        
        # T8: A30 B01 -> C31
        U_e[4, 1] = 1
        V_e[1, 1] = 1
        W_e[5, 1] = 1
        
        # T9: A31 B10 -> C30
        U_e[5, 2] = 1
        V_e[2, 2] = 1
        W_e[2, 2] = 1
        
        # T10: A31 B11 -> C31
        U_e[5, 3] = 1
        V_e[3, 3] = 1
        W_e[5, 3] = 1
        
        # Combine Strassen (7) and Extra (4)
        U_total = np.zeros((6, 11))
        V_total = np.zeros((4, 11))
        W_total = np.zeros((6, 11))
        
        # Strassen part
        # U_s is 4x7. Map to 6x7.
        U_total[0:4, 0:7] = U_s
        V_total[0:4, 0:7] = V_s
        W_total[0:4, 0:7] = W_s # W_s is 4x7, map to 6x7 (rows 0-3)
        
        # Extra part
        U_total[:, 7:11] = U_e
        V_total[:, 7:11] = V_e
        W_total[:, 7:11] = W_e
        
        return U_total, V_total, W_total, 11

    U, V, W, rank = build_decomp(n, m, p)
    
    def to_list(mat):
        return mat.tolist()
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
