import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition
    # U, V, W are 4x7 matrices (flattened 2x2)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Reconstruction
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U (A side), V (B side), W (C side)
    # Flattened indices: 0:00, 1:01, 2:10, 3:11
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C00, C11 parts? No, W is for C indices.
    # Wait, W2 is the coefficient for the product M_t in the reconstruction of the 2x2 product.
    # For 2x2x2, C_pq = sum_t W2[pq, t] * M_t.
    # So W2 should be 4x7 where rows are C indices (00, 01, 10, 11).
    
    # Let's define W2 properly.
    W2[0, 0] = 1 # C00 gets M1
    W2[3, 0] = 1 # C11 gets M1
    
    W2[1, 1] = 0 # C01
    W2[2, 1] = 1 # C10 gets M2
    W2[3, 1] = -1 # C11 gets -M2
    
    W2[1, 2] = 1 # C01 gets M3
    W2[3, 2] = 1 # C11 gets M3
    
    W2[0, 3] = 1 # C00 gets M4
    W2[2, 3] = 1 # C10 gets M4
    
    W2[0, 4] = -1 # C00 gets -M5
    W2[1, 4] = 1 # C01 gets M5
    
    W2[3, 5] = 1 # C11 gets M6
    
    W2[0, 6] = 1 # C00 gets M7
    
    # Now U2 and V2
    # M1: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    
    # M2: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    
    # M3: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    
    # M4: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    
    # M5: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    
    # Now build 4x4x4 decomposition
    # Rank 49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Macro terms for 4x4
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Coefficients for A blocks (00, 01, 10, 11)
    # Block 00 is indices 0-3, 01 is 4-7, 10 is 8-11, 11 is 12-15
    A_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    B_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Reconstruction coefficients for 4x4
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # C blocks: 00 (0-3), 01 (4-7), 10 (8-11), 11 (12-15)
    
    C_coeffs = [
        [1, 0, 0, 0, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0], # C01
        [0, 1, 0, 1, 0, 0, 0], # C10
        [1, -1, 1, 0, 0, 1, 0] # C11
    ]
    
    for s in range(7):
        # For each macro term s, we have 7 micro terms t
        for t in range(7):
            r = s * 7 + t
            
            # U part
            # U4[i, r] = sum_{block} A_coeffs[s][block] * U2[local_i, t]
            for block in range(4):
                if A_coeffs[s][block] != 0:
                    offset = block * 4
                    for local_i in range(4):
                        U4[offset + local_i, r] += A_coeffs[s][block] * U2[local_i, t]
                        
            # V part
            for block in range(4):
                if B_coeffs[s][block] != 0:
                    offset = block * 4
                    for local_i in range(4):
                        V4[offset + local_i, r] += B_coeffs[s][block] * V2[local_i, t]
                        
            # W part
            # W4[k, r] = sum_{c_block} C_coeffs[c_block][s] * W2[local_k, t]
            for c_block in range(4):
                if C_coeffs[c_block][s] != 0:
                    offset = c_block * 4
                    for local_k in range(4):
                        W4[offset + local_k, r] += C_coeffs[c_block][s] * W2[local_k, t]
                        
    # Now extend to 4x4x8
    # Rank 98
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # First half (cols 0-3 of B, cols 0-3 of C)
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    # Second half (cols 4-7 of B, cols 4-7 of C)
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4
    W[16:32, 49:98] = W4
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    construct()
