# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 2, 3, 3
    
    # Rank 17 decomposition
    # Partition:
    # A: 2x3 -> [A_2x2, A_2x1]
    # B: 3x3 -> [B_2x3; B_1x3]
    # C = A_2x2 B_2x2 + A_2x2 B_2x1 + A_2x1 B_1x2 + A_2x1 B_1x1 ?
    # No, B is 3x3.
    # B = [B_2x2, B_2x1; B_1x2, B_1x1]
    # C = A_2x2 B_2x2 + A_2x2 B_2x1 + A_2x1 B_1x2 + A_2x1 B_1x1
    # Dimensions:
    # A_2x2: 2x2
    # A_2x1: 2x1
    # B_2x2: 2x2
    # B_2x1: 2x1
    # B_1x2: 1x2
    # B_1x1: 1x1
    
    # Terms:
    # 1. A_2x2 * B_2x2 (2x2x2) -> Rank 7 (Strassen)
    # 2. A_2x2 * B_2x1 (2x2x1) -> Rank 4 (Schoolbook)
    # 3. A_2x1 * B_1x2 (2x1x2) -> Rank 4 (Schoolbook)
    # 4. A_2x1 * B_1x1 (2x1x1) -> Rank 2 (Schoolbook)
    
    # Total Rank: 7 + 4 + 4 + 2 = 17.
    
    U = np.zeros((6, 17))
    V = np.zeros((9, 17))
    W = np.zeros((6, 17))
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val, col):
        U[u_idx, col] = u_val
        V[v_idx, col] = v_val
        W[w_idx, col] = w_val

    # --- Part 1: A_2x2 * B_2x2 (Strassen) ---
    # Indices in U: i in {0,1}, j in {0,1} -> i*3 + j
    # Indices in V: j in {0,1}, k in {0,1} -> j*3 + k
    # Indices in W: k in {0,1}, i in {0,1} -> k*2 + i
    
    # Strassen for 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    col = 0
    # M1
    U[0, col] = 1; U[3, col] = 1 # A00, A11 (i=0,j=0; i=1,j=1) -> idx 0, 3
    V[0, col] = 1; V[4, col] = 1 # B00, B11 (j=0,k=0; j=1,k=1) -> idx 0, 4
    W[0, col] = 1; W[1, col] = -1; W[2, col] = 1; W[3, col] = 1 # C00, C11, C01, C10?
    # W[k*2+i]. C00 (k=0,i=0), C11 (k=1,i=1), C01 (k=1,i=0), C10 (k=0,i=1)
    # M1 contributes to C00 (+1), C11 (+1), C01 (-1), C10 (+1)?
    # Formula: C11=M1+M4-M5+M7. C12=M3+M5. C21=M2+M4. C22=M1-M2+M3+M6.
    # W indices:
    # C00 (i=0,k=0): W[0*2+0] = W[0]
    # C01 (i=0,k=1): W[1*2+0] = W[2]
    # C10 (i=1,k=0): W[0*2+1] = W[1]
    # C11 (i=1,k=1): W[1*2+1] = W[3]
    
    # M1: C00+1, C11+1, C01-1, C10+1? No.
    # C11 (top-left) = M1 + M4 - M5 + M7
    # C12 (top-right) = M3 + M5
    # C21 (bottom-left) = M2 + M4
    # C22 (bottom-right) = M1 - M2 + M3 + M6
    
    # So:
    # M1: C11(+1), C22(+1)
    # M2: C21(+1), C22(-1)
    # M3: C12(+1), C22(+1)
    # M4: C11(+1), C21(+1)
    # M5: C11(-1), C12(+1)
    # M6: C22(+1)
    # M7: C11(+1)
    
    # Let's map to W[k*2+i]:
    # C11 (i=0,k=0) -> W[0]
    # C12 (i=0,k=1) -> W[2]
    # C21 (i=1,k=0) -> W[1]
    # C22 (i=1,k=1) -> W[3]
    
    # M1: W[0]=1, W[3]=1
    # M2: W[1]=1, W[3]=-1
    # M3: W[2]=1, W[3]=1
    # M4: W[0]=1, W[1]=1
    # M5: W[0]=-1, W[2]=1
    # M6: W[3]=1
    # M7: W[0]=1
    
    # U/V for M1
    U[0, col] = 1; U[3, col] = 1
    V[0, col] = 1; V[4, col] = 1
    W[0, col] = 1; W[3, col] = 1
    col += 1
    
    # U/V for M2
    U[3, col] = 1; U[4, col] = 1 # A10, A11
    V[0, col] = 1 # B00
    W[1, col] = 1; W[3, col] = -1
    col += 1
    
    # U/V for M3
    U[0, col] = 1 # A00
    V[1, col] = 1; V[4, col] = -1 # B01, B11
    W[2, col] = 1; W[3, col] = 1
    col += 1
    
    # U/V for M4
    U[3, col] = 1 # A11
    V[3, col] = 1; V[0, col] = -1 # B10, B00
    W[0, col] = 1; W[1, col] = 1
    col += 1
    
    # U/V for M5
    U[0, col] = 1; U[1, col] = 1 # A00, A01
    V[4, col] = 1 # B11
    W[0, col] = -1; W[2, col] = 1
    col += 1
    
    # U/V for M6
    U[3, col] = 1; U[0, col] = -1 # A10, A00
    V[0, col] = 1; V[1, col] = 1 # B00, B01
    W[3, col] = 1
    col += 1
    
    # U/V for M7
    U[1, col] = 1; U[3, col] = -1 # A01, A11
    V[3, col] = 1; V[4, col] = 1 # B10, B11
    W[0, col] = 1
    col += 1
    
    # --- Part 2: A_2x2 * B_2x1 ---
    # A_2x2: i in {0,1}, j in {0,1}
    # B_2x1: j in {0,1}, k=2
    # C: i in {0,1}, k=2
    # Schoolbook rank 4.
    
    for i in range(2):
        for j in range(2):
            U[i*3 + j, col] = 1
            V[j*3 + 2, col] = 1
            W[2*2 + i, col] = 1 # k=2, i
            col += 1
            
    # --- Part 3: A_2x1 * B_1x2 ---
    # A_2x1: i in {0,1}, j=2
    # B_1x2: j=2, k in {0,1}
    # C: i in {0,1}, k in {0,1}
    # Schoolbook rank 4.
    
    for i in range(2):
        for k in range(2):
            U[i*3 + 2, col] = 1
            V[2*3 + k, col] = 1
            W[k*2 + i, col] = 1
            col += 1
            
    # --- Part 4: A_2x1 * B_1x1 ---
    # A_2x1: i in {0,1}, j=2
    # B_1x1: j=2, k=2
    # C: i in {0,1}, k=2
    # Schoolbook rank 2.
    
    for i in range(2):
        U[i*3 + 2, col] = 1
        V[2*3 + 2, col] = 1
        W[2*2 + i, col] = 1
        col += 1
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
