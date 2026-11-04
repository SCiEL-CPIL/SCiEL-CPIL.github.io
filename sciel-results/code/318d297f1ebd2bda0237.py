import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Strassen 2x2x2 decomposition
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((4, 7))
    W_strassen = np.zeros((4, 7))
    
    # r=0: (A00+A11)(B00+B11)
    U_strassen[0, 0] = 1  # A00
    U_strassen[3, 0] = 1  # A11
    V_strassen[0, 0] = 1  # B00
    V_strassen[3, 0] = 1  # B11
    W_strassen[0, 0] = 1  # C00
    W_strassen[3, 0] = -1 # C11 (will add M4, M5, M6 later)
    # Actually let me be more careful. Let me define the 7 terms properly.
    
    # Let me restart with clear definitions.
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B10-B11)
    # M3 = A11(B01-B10)
    # M4 = (A00+A10)B11
    # M5 = (A01+A11)(B00+B10)
    # M6 = (A01-A10)(B11+B01)
    
    # C00 = M0 + M3 - M5 + M6
    # C01 = M2 + M5
    # C10 = M1 - M4 + M6
    # C11 = M0 - M1 + M2 + M4
    
    # For each term r, U[i,j,r] is coefficient of A[i,j], V[j,k,r] is coefficient of B[j,k],
    # W[k,i,r] is coefficient of C[i,k]
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M0 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1
    U_s[0, 0] = 1  # A00
    U_s[3, 0] = 1  # A11
    # V: B00=1, B11=1
    V_s[0, 0] = 1  # B00
    V_s[3, 0] = 1  # B11
    # W: C00=1, C01=-1, C10=-1, C11=1
    W_s[0, 0] = 1   # C00
    W_s[1, 0] = -1  # C01
    W_s[2, 0] = -1  # C10
    W_s[3, 0] = 1   # C11
    
    # M1 = (A10+A11)B00
    U_s[2, 1] = 1  # A10
    U_s[3, 1] = 1  # A11
    V_s[0, 1] = 1  # B00
    W_s[2, 1] = 1   # C10
    W_s[3, 1] = -1  # C11
    
    # M2 = A00(B10-B11)
    U_s[0, 2] = 1  # A00
    V_s[2, 2] = 1  # B10
    V_s[3, 2] = -1 # B11
    W_s[1, 2] = 1   # C01
    W_s[3, 2] = 1   # C11
    
    # M3 = A11(B01-B10)
    U_s[3, 3] = 1  # A11
    V_s[1, 3] = 1  # B01
    V_s[2, 3] = -1 # B10
    W_s[0, 3] = 1   # C00
    
    # M4 = (A00+A10)B11
    U_s[0, 4] = 1  # A00
    U_s[2, 4] = 1  # A10
    V_s[3, 4] = 1  # B11
    W_s[2, 4] = -1  # C10
    W_s[3, 4] = 1   # C11
    
    # M5 = (A01+A11)(B00+B10)
    U_s[1, 5] = 1  # A01
    U_s[3, 5] = 1  # A11
    V_s[0, 5] = 1  # B00
    V_s[2, 5] = 1  # B10
    W_s[0, 5] = -1  # C00
    W_s[1, 5] = 1   # C01
    
    # M6 = (A01-A10)(B11+B01)
    U_s[1, 6] = 1  # A01
    U_s[2, 6] = -1 # A10
    V_s[3, 6] = 1  # B11
    V_s[1, 6] = 1  # B01
    W_s[0, 6] = 1   # C00
    W_s[2, 6] = 1   # C10
    
    # Schoolbook 2x1x2 decomposition (rank 4)
    # U: (4, 4), V: (2, 4), W: (4, 4)
    U_sb = np.zeros((4, 4))
    V_sb = np.zeros((2, 4))
    W_sb = np.zeros((4, 4))
    
    # Terms: A00*B00, A00*B01, A10*B00, A10*B01
    U_sb[0, 0] = 1  # A00
    U_sb[0, 1] = 1  # A00
    U_sb[1, 2] = 1  # A10
    U_sb[1, 3] = 1  # A10
    
    V_sb[0, 0] = 1  # B00
    V_sb[1, 1] = 1  # B01
    V_sb[0, 2] = 1  # B00
    V_sb[1, 3] = 1  # B01
    
    W_sb[0, 0] = 1  # C00
    W_sb[1, 1] = 1  # C01
    W_sb[2, 2] = 1  # C10
    W_sb[3, 3] = 1  # C11
    
    # Now build the global matrices
    # n=2, m=5, p=6
    # U shape: (n*m, rank) = (10, rank)
    # V shape: (m*p, rank) = (30, rank)
    # W shape: (p*n, rank) = (12, rank)
    
    # Block structure:
    # For j in {0,1} and k in {0,1}, {2,3}, {4,5}: use Strassen (7 terms each)
    # For j in {2,3} and k in {0,1}, {2,3}, {4,5}: use Strassen (7 terms each)
    # For j=4 and k in {0,1}, {2,3}, {4,5}: use schoolbook 2x1x2 (4 terms each)
    
    # Total rank = 6*7 + 3*4 = 42 + 12 = 54
    
    total_rank = 54
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    rank_offset = 0
    
    # 6 Strassen blocks
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            for r in range(7):
                for i in range(2):
                    for j_l in range(2):
                        U[i * m + (j_start + j_l), rank_offset] = U_s[i * 2 + j_l, r]
                for j_l in range(2):
                    for k_l in range(2):
                        V[(j_start + j_l) * p + (k_start + k_l), rank_offset] = V_s[j_l * 2 + k_l, r]
                for i in range(2):
                    for k_l in range(2):
                        W[(k_start + k_l) * n + i, rank_offset] = W_s[k_l * 2 + i, r]
                rank_offset += 1
    
    # 3 schoolbook blocks for j=4
    for k_start in [0, 2, 4]:
        for r in range(4):
            for i in range(2):
                U[i * m + 4, rank_offset] = U_sb[i * 2, r]
            for k_l in range(2):
                V[4 * p + (k_start + k_l), rank_offset] = V_sb[k_l, r]
            for i in range(2):
                for k_l in range(2):
                    W[(k_start + k_l) * n + i, rank_offset] = W_sb[k_l * 2 + i, r]
            rank_offset += 1
    
    # Convert to lists of lists with integer/half-integer entries
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

