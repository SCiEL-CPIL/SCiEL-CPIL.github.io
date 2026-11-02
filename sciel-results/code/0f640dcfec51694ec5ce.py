"""Matrix multiplication 6,6,6 with Kronecker product structure."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # U2: shape (4, 7), V2: shape (4, 7), W2: shape (4, 7)
    # Convention: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # P1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # P2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    
    # P3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1
    
    # P4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # P5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[3, 4] = 1; W2[0, 4] = -1
    
    # P6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # P7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Schoolbook for 3x3 (rank 27) for guaranteed correctness
    U3 = np.zeros((9, 27), dtype=int)
    V3 = np.zeros((9, 27), dtype=int)
    W3 = np.zeros((9, 27), dtype=int)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j, col] = 1
                V3[j*3+k, col] = 1
                W3[k*3+i, col] = 1
                col += 1
    
    # Construct Kronecker product: T_6 = T_2 ⊗ T_3
    # rank = 7 * 27 = 189
    rank = 7 * 27
    U6 = np.zeros((36, rank), dtype=int)
    V6 = np.zeros((36, rank), dtype=int)
    W6 = np.zeros((36, rank), dtype=int)
    
    for r in range(7):
        for s in range(27):
            idx = r * 27 + s
            
            # Get nonzero patterns
            u2_col = U2[:, r]
            v2_col = V2[:, r]
            w2_col = W2[:, r]
            u3_col = U3[:, s]
            v3_col = V3[:, s]
            w3_col = W3[:, s]
            
            # For each combination of nonzeros
            for a2_idx in range(4):
                if u2_col[a2_idx] == 0:
                    continue
                a2_i = a2_idx // 2
                a2_j = a2_idx % 2
                
                for b2_idx in range(4):
                    if v2_col[b2_idx] == 0:
                        continue
                    b2_j = b2_idx // 2
                    b2_k = b2_idx % 2
                    
                    for a3_idx in range(9):
                        if u3_col[a3_idx] == 0:
                            continue
                        a3_i = a3_idx // 3
                        a3_j = a3_idx % 3
                        
                        # Global A index
                        a_i = a2_i * 3 + a3_i
                        a_j = a2_j * 3 + a3_j
                        a_global = a_i * 6 + a_j
                        
                        for b3_idx in range(9):
                            if v3_col[b3_idx] == 0:
                                continue
                            b3_j = b3_idx // 3
                            b3_k = b3_idx % 3
                            
                            # Global B index
                            b_j = b2_j * 3 + b3_j
                            b_k = b2_k * 3 + b3_k
                            b_global = b_j * 6 + b_k
                            
                            # Set U6 and V6
                            U6[a_global, idx] = u2_col[a2_idx] * u3_col[a3_idx]
                            V6[b_global, idx] = v2_col[b2_idx] * v3_col[b3_idx]
                            
                            # Set W6
                            for c2_idx in range(4):
                                if w2_col[c2_idx] == 0:
                                    continue
                                c2_k = c2_idx // 2
                                c2_i = c2_idx % 2
                                
                                for c3_idx in range(9):
                                    if w3_col[c3_idx] == 0:
                                        continue
                                    c3_k = c3_idx // 3
                                    c3_i = c3_idx % 3
                                    
                                    # Global C index (k*n+i convention)
                                    c_k = c2_k * 3 + c3_k
                                    c_i = c2_i * 3 + c3_i
                                    c_global = c_k * 6 + c_i
                                    
                                    W6[c_global, idx] = w2_col[c2_idx] * w3_col[c3_idx]
    
    # Convert to lists
    U_list = U6.tolist()
    V_list = V6.tolist()
    W_list = W6.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
