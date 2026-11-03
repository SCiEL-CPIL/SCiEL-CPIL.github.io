import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Rank 49 Strassen for 4x4x4
    # Outer Strassen 7 terms
    # Inner Strassen 7 terms
    
    # Define inner Strassen matrices for 2x2x2
    # Indices: U (4x7), V (4x7), W (4x7)
    # Rows for U: 00, 01, 10, 11
    # Rows for V: 00, 01, 10, 11
    # Rows for W: 00, 01, 10, 11
    
    U_inner = np.zeros((4, 7))
    V_inner = np.zeros((4, 7))
    W_inner = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B11) -> C00, C11
    U_inner[0, 0] = 1; U_inner[3, 0] = 1
    V_inner[0, 0] = 1; V_inner[3, 0] = 1
    W_inner[0, 0] = 1; W_inner[3, 0] = 1
    
    # Term 2: (A10+A11)B00 -> C10
    U_inner[2, 1] = 1; U_inner[3, 1] = 1
    V_inner[0, 1] = 1
    W_inner[2, 1] = 1
    
    # Term 3: A00(B01-B11) -> C01, C11
    U_inner[0, 2] = 1
    V_inner[1, 2] = 1; V_inner[3, 2] = -1
    W_inner[1, 2] = 1; W_inner[3, 2] = 1
    
    # Term 4: A11(B10-B00) -> C10, C00
    U_inner[3, 3] = 1
    V_inner[2, 3] = 1; V_inner[0, 3] = -1
    W_inner[2, 3] = 1; W_inner[0, 3] = 1
    
    # Term 5: (A00+A01)B11 -> C01, C00(-)
    U_inner[0, 4] = 1; U_inner[1, 4] = 1
    V_inner[3, 4] = 1
    W_inner[1, 4] = 1; W_inner[0, 4] = -1
    
    # Term 6: (A10-A00)(B00+B01) -> C11
    U_inner[2, 5] = 1; U_inner[0, 5] = -1
    V_inner[0, 5] = 1; V_inner[1, 5] = 1
    W_inner[3, 5] = 1
    
    # Term 7: (A01-A11)(B10+B11) -> C00, C11(-)
    U_inner[1, 6] = 1; U_inner[3, 6] = -1
    V_inner[2, 6] = 1; V_inner[3, 6] = 1
    W_inner[0, 6] = 1; W_inner[3, 6] = -1
    
    # Define outer Strassen matrices for 4x4x4 (blocks)
    # Rows for U_outer: 00, 01, 10, 11 (blocks)
    # Rows for V_outer: 00, 01, 10, 11 (blocks)
    # Rows for W_outer: 00, 01, 10, 11 (blocks)
    
    U_outer = np.zeros((4, 7))
    V_outer = np.zeros((4, 7))
    W_outer = np.zeros((4, 7))
    
    # Same structure as inner
    U_outer[0, 0] = 1; U_outer[3, 0] = 1
    V_outer[0, 0] = 1; V_outer[3, 0] = 1
    W_outer[0, 0] = 1; W_outer[3, 0] = 1
    
    U_outer[2, 1] = 1; U_outer[3, 1] = 1
    V_outer[0, 1] = 1
    W_outer[2, 1] = 1
    
    U_outer[0, 2] = 1
    V_outer[1, 2] = 1; V_outer[3, 2] = -1
    W_outer[1, 2] = 1; W_outer[3, 2] = 1
    
    U_outer[3, 3] = 1
    V_outer[2, 3] = 1; V_outer[0, 3] = -1
    W_outer[2, 3] = 1; W_outer[0, 3] = 1
    
    U_outer[0, 4] = 1; U_outer[1, 4] = 1
    V_outer[3, 4] = 1
    W_outer[1, 4] = 1; W_outer[0, 4] = -1
    
    U_outer[2, 5] = 1; U_outer[0, 5] = -1
    V_outer[0, 5] = 1; V_outer[1, 5] = 1
    W_outer[3, 5] = 1
    
    U_outer[1, 6] = 1; U_outer[3, 6] = -1
    V_outer[2, 6] = 1; V_outer[3, 6] = 1
    W_outer[0, 6] = 1; W_outer[3, 6] = -1
    
    # Construct global U, V, W for rank 49
    rank_49 = 49
    U_49 = np.zeros((16, rank_49))
    V_49 = np.zeros((16, rank_49))
    W_49 = np.zeros((16, rank_49))
    
    for t in range(7):
        for u in range(7):
            c = t * 7 + u
            for I in range(2):
                for J in range(2):
                    for i in range(2):
                        for j in range(2):
                            # Global A index: 2I+i, 2J+j -> row 2I+i * 4 + 2J+j
                            row_a = (2*I + i) * 4 + (2*J + j)
                            # Outer block index: I, J
                            u_val_outer = U_outer[I*2+J, t]
                            # Inner element index: i, j
                            u_val_inner = U_inner[i*2+j, u]
                            U_49[row_a, c] = u_val_outer * u_val_inner
                            
                            # Global B index: 2J+j, 2K+k
                            # We need to iterate K, k for V
                            # But V_outer and V_inner are defined for specific blocks/elements
                            # Let's fill V and W in separate loops or combined
                            
            # Fill V_49
            for J in range(2):
                for K in range(2):
                    for j in range(2):
                        for k in range(2):
                            row_b = (2*J + j) * 4 + (2*K + k)
                            v_val_outer = V_outer[J*2+K, t]
                            v_val_inner = V_inner[j*2+k, u]
                            V_49[row_b, c] = v_val_outer * v_val_inner
                            
            # Fill W_49
            for I in range(2):
                for K in range(2):
                    for i in range(2):
                        for k in range(2):
                            row_c = (2*K + k) * 4 + (2*I + i) # W[k*n+i] -> k*4+i
                            # Wait, W index is k*n+i. n=4.
                            # k is 0..3. i is 0..3.
                            # In our block decomposition:
                            # Global C row: 2K+k. Col: 2I+i.
                            # W index: (2K+k)*4 + (2I+i).
                            w_val_outer = W_outer[I*2+K, t]
                            w_val_inner = W_inner[i*2+k, u]
                            W_49[row_c, c] = w_val_outer * w_val_inner

    # Add 16 terms for the 5th column
    rank_total = rank_49 + 16
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total))
    W = np.zeros((20, rank_total))
    
    # Copy first 49
    U[:, :49] = U_49
    V[:16, :49] = V_49
    W[:16, :49] = W_49
    
    # Add 16 terms
    for idx in range(16):
        i = idx // 4
        j = idx % 4
        c = 49 + idx
        U[i*4+j, c] = 1
        V[j*5+4, c] = 1
        W[4*4+i, c] = 1 # W[k*n+i] with k=4, n=4 -> 16+i
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
