import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Part 1: A_1 (2x1) * B_1 (1x7)
    # Rank 14
    U1 = np.zeros((2, 14)) # i, r
    V1 = np.zeros((7, 14)) # k, r
    W1 = np.zeros((14, 14)) # (k,i), r -> Wait, W shape is (p*n, rank) = (14, 14)
    
    # Mapping for Part 1:
    # i in 0..1, j=0, k in 0..6
    # U row index: i*3 + 0
    # V row index: 0*7 + k = k
    # W row index: k*2 + i
    
    r = 0
    for i in range(2):
        for k in range(7):
            U1[i, r] = 1
            V1[k, r] = 1
            W1[k*2 + i, r] = 1
            r += 1
            
    # Part 2: A_2 (2x2) * B_2 (2x7)
    # We split B_2 into B_2a (2x3) and B_2b (2x4)
    # B_2a uses k in 0..2, B_2b uses k in 3..6
    
    # Sub-part 2a: M_{2,2,3}
    # Split B_2a into B_2a1 (2x2, k=0,1) and B_2a2 (2x1, k=2)
    # Rank 7 (Strassen) + 4 (Schoolbook) = 11
    
    # Sub-part 2b: M_{2,2,4}
    # Split B_2b into B_2b1 (2x2, k=3,4) and B_2b2 (2x2, k=5,6)
    # Rank 7 + 7 = 14
    
    # Total Part 2 Rank: 11 + 14 = 25
    
    U2 = np.zeros((2*2, 25)) # i, j_rel (0,1) -> Global j=1,2
    V2 = np.zeros((2*7, 25)) # j_rel, k
    W2 = np.zeros((14, 25))  # k, i
    
    # Helper to add Strassen block
    def add_strassen(U_block, V_block, W_block, i_offset, j_offset, k_offset):
        # Strassen for 2x2x2
        # U indices: (i, j)
        # V indices: (j, k)
        # W indices: (k, i)
        # 7 multiplications
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        
        # Coefficients for U (A matrix)
        # U[i, j, r]
        # Coefficients for V (B matrix)
        # V[j, k, r]
        # Coefficients for W (C matrix)
        # W[k, i, r]
        
        # We will fill columns of U_block, V_block, W_block
        # U_block shape (4, rank_chunk)
        # V_block shape (4, rank_chunk) -- Note: V_block here is local 2x2, needs mapping to global 2x7
        # W_block shape (4, rank_chunk) -- Local 2x2, needs mapping to global 14
        
        pass # Implemented below inline for clarity

    # Let's build the 25 columns for Part 2
    col = 0
    
    # 1. Strassen for k=0,1 (j_rel 0,1)
    # Local i: 0,1. Local j: 0,1. Local k: 0,1.
    # Global i: 0,1. Global j: 1,2. Global k: 0,1.
    
    # Strassen Coeffs
    # r=0: M1
    U2[0*2+0, col] = 1; U2[1*2+1, col] = 1 # A00, A11
    V2[0*2+0, col] = 1; V2[1*2+1, col] = 1 # B00, B11 (Local k 0,1)
    W2[0*2+0, col] = 1; W2[1*2+1, col] = 1; W2[0*2+1, col] = 1; W2[1*2+0, col] = 1 # C00, C11, C01, C10
    col += 1
    
    # r=1: M2
    U2[1*2+0, col] = 1; U2[1*2+1, col] = 1 # A10, A11
    V2[0*2+0, col] = 1 # B00
    W2[1*2+1, col] = 1; W2[1*2+0, col] = 1 # C11, C10
    col += 1
    
    # r=2: M3
    U2[0*2+0, col] = 1 # A00
    V2[1*2+0, col] = 1; V2[1*2+1, col] = -1 # B10, B11
    W2[0*2+1, col] = 1 # C01
    col += 1
    
    # r=3: M4
    U2[1*2+1, col] = 1 # A11
    V2[0*2+1, col] = 1; V2[0*2+0, col] = -1 # B01, B00
    W2[1*2+0, col] = 1 # C10
    col += 1
    
    # r=4: M5
    U2[0*2+0, col] = 1; U2[0*2+1, col] = 1 # A00, A01
    V2[1*2+1, col] = 1 # B11
    W2[0*2+0, col] = 1; W2[1*2+0, col] = 1 # C00, C10
    col += 1
    
    # r=5: M6
    U2[1*2+0, col] = 1; U2[0*2+0, col] = -1 # A10, A00
    V2[0*2+0, col] = 1; V2[0*2+1, col] = 1 # B00, B01
    W2[1*2+1, col] = 1; W2[0*2+1, col] = 1 # C11, C01
    col += 1
    
    # r=6: M7
    U2[0*2+1, col] = 1; U2[1*2+1, col] = -1 # A01, A11
    V2[0*2+1, col] = 1; V2[1*2+1, col] = 1 # B01, B11
    W2[0*2+0, col] = 1; W2[1*2+1, col] = 1 # C00, C11
    col += 1
    
    # 2. Schoolbook for k=2 (j_rel 0,1)
    # Local i: 0,1. Local j: 0,1. Local k: 0 (which is global k=2)
    # Global k=2.
    
    # j_rel=0 (Global j=1)
    U2[0*2+0, col] = 1; V2[0*2+2, col] = 1; W2[2*2+0, col] = 1
    col += 1
    U2[1*2+0, col] = 1; V2[0*2+2, col] = 1; W2[2*2+1, col] = 1
    col += 1
    # j_rel=1 (Global j=2)
    U2[0*2+1, col] = 1; V2[1*2+2, col] = 1; W2[2*2+0, col] = 1
    col += 1
    U2[1*2+1, col] = 1; V2[1*2+2, col] = 1; W2[2*2+1, col] = 1
    col += 1
    
    # 3. Strassen for k=3,4
    # Local k 0,1 -> Global k 3,4
    
    # r=7..13 (7 terms)
    # M1
    U2[0*2+0, col] = 1; U2[1*2+1, col] = 1
    V2[0*2+3, col] = 1; V2[1*2+4, col] = 1
    W2[3*2+0, col] = 1; W2[4*2+1, col] = 1; W2[3*2+1, col] = 1; W2[4*2+0, col] = 1
    col += 1
    # M2
    U2[1*2+0, col] = 1; U2[1*2+1, col] = 1
    V2[0*2+3, col] = 1
    W2[4*2+1, col] = 1; W2[4*2+0, col] = 1
    col += 1
    # M3
    U2[0*2+0, col] = 1
    V2[1*2+3, col] = 1; V2[1*2+4, col] = -1
    W2[3*2+1, col] = 1
    col += 1
    # M4
    U2[1*2+1, col] = 1
    V2[0*2+4, col] = 1; V2[0*2+3, col] = -1
    W2[4*2+0, col] = 1
    col += 1
    # M5
    U2[0*2+0, col] = 1; U2[0*2+1, col] = 1
    V2[1*2+4, col] = 1
    W2[3*2+0, col] = 1; W2[4*2+0, col] = 1
    col += 1
    # M6
    U2[1*2+0, col] = 1; U2[0*2+0, col] = -1
    V2[0*2+3, col] = 1; V2[0*2+4, col] = 1
    W2[4*2+1, col] = 1; W2[3*2+1, col] = 1
    col += 1
    # M7
    U2[0*2+1, col] = 1; U2[1*2+1, col] = -1
    V2[0*2+4, col] = 1; V2[1*2+4, col] = 1
    W2[3*2+0, col] = 1; W2[4*2+1, col] = 1
    col += 1
    
    # 4. Strassen for k=5,6
    # Local k 0,1 -> Global k 5,6
    
    # r=14..20
    U2[0*2+0, col] = 1; U2[1*2+1, col] = 1
    V2[0*2+5, col] = 1; V2[1*2+6, col] = 1
    W2[5*2+0, col] = 1; W2[6*2+1, col] = 1; W2[5*2+1, col] = 1; W2[6*2+0, col] = 1
    col += 1
    U2[1*2+0, col] = 1; U2[1*2+1, col] = 1
    V2[0*2+5, col] = 1
    W2[6*2+1, col] = 1; W2[6*2+0, col] = 1
    col += 1
    U2[0*2+0, col] = 1
    V2[1*2+5, col] = 1; V2[1*2+6, col] = -1
    W2[5*2+1, col] = 1
    col += 1
    U2[1*2+1, col] = 1
    V2[0*2+6, col] = 1; V2[0*2+5, col] = -1
    W2[6*2+0, col] = 1
    col += 1
    U2[0*2+0, col] = 1; U2[0*2+1, col] = 1
    V2[1*2+6, col] = 1
    W2[5*2+0, col] = 1; W2[6*2+0, col] = 1
    col += 1
    U2[1*2+0, col] = 1; U2[0*2+0, col] = -1
    V2[0*2+5, col] = 1; V2[0*2+6, col] = 1
    W2[6*2+1, col] = 1; W2[5*2+1, col] = 1
    col += 1
    U2[0*2+1, col] = 1; U2[1*2+1, col] = -1
    V2[0*2+6, col] = 1; V2[1*2+6, col] = 1
    W2[5*2+0, col] = 1; W2[6*2+1, col] = 1
    col += 1
    
    # Combine
    # U_full shape (6, 39)
    # V_full shape (21, 39)
    # W_full shape (14, 39)
    
    U_full = np.zeros((6, 39))
    V_full = np.zeros((21, 39))
    W_full = np.zeros((14, 39))
    
    # Part 1 (14 cols)
    for i in range(2):
        U_full[i*3 + 0, 0:14] = U1[i, :]
    for k in range(7):
        V_full[0*7 + k, 0:14] = V1[k, :]
    W_full[:, 0:14] = W1
    
    # Part 2 (25 cols)
    # U2 rows: i*2 + j_rel. Global j = j_rel + 1.
    # Global row = i*3 + (j_rel+1)
    for i in range(2):
        for j_rel in range(2):
            U_full[i*3 + (j_rel+1), 14:39] = U2[i*2 + j_rel, :]
            
    # V2 rows: j_rel*2 + k. Global j = j_rel + 1.
    # Global row = (j_rel+1)*7 + k
    for j_rel in range(2):
        for k in range(7):
            V_full[(j_rel+1)*7 + k, 14:39] = V2[j_rel*2 + k, :]
            
    W_full[:, 14:39] = W2
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
