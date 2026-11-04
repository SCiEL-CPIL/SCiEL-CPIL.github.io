# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a tensor decomposition for 3x3 * 3x6 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    # Strassen's algorithm for 3x3 matrix multiplication
    # C = A * B where A, B, C are 3x3
    # The 7 products are:
    # M1 = (A00 + A22) * (B00 + B22)
    # M2 = (A20 + A22) * B00
    # M3 = A00 * (B01 - B21)
    # M4 = A22 * (B11 - B01)
    # M5 = (A01 + A11) * B11
    # M6 = (A10 + A20) * (B00 + B10)
    # M7 = (A11 + A22) * (B11 + B21)
    #
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    # C20 = M6 - M4 + M7
    # C21 = M2 + M6
    # C22 = M1 - M3 + M6
    
    # Strassen decomposition for one 3x3x3 block:
    # Each of the 7 terms has U, V, W vectors of length 9 (for 3x3 case)
    # Indices in 3x3 case:
    # a = i*3 + j
    # b = j*3 + k  (k in 0..2)
    # c = k*3 + i  (k in 0..2)
    
    strassen_terms = []
    
    # Term 1: M1 = (A00 + A22) * (B00 + B22)
    U1 = [0]*9; U1[0] = 1; U1[8] = 1
    V1 = [0]*9; V1[0] = 1; V1[8] = 1
    W1 = [0]*9; W1[0] = 1; W1[4] = 1; W1[8] = 1
    strassen_terms.append((U1, V1, W1))
    
    # Term 2: M2 = (A20 + A22) * B00
    U2 = [0]*9; U2[6] = 1; U2[8] = 1
    V2 = [0]*9; V2[0] = 1
    W2 = [0]*9; W2[3] = 1; W2[7] = 1
    strassen_terms.append((U2, V2, W2))
    
    # Term 3: M3 = A00 * (B01 - B21)
    U3 = [0]*9; U3[0] = 1
    V3 = [0]*9; V3[1] = 1; V3[7] = -1
    W3 = [0]*9; W3[1] = 1; W3[4] = 1; W3[8] = -1
    strassen_terms.append((U3, V3, W3))
    
    # Term 4: M4 = A22 * (B11 - B01)
    U4 = [0]*9; U4[8] = 1
    V4 = [0]*9; V4[4] = 1; V4[1] = -1
    W4 = [0]*9; W4[0] = -1; W4[3] = 1; W4[6] = -1
    strassen_terms.append((U4, V4, W4))
    
    # Term 5: M5 = (A01 + A11) * B11
    U5 = [0]*9; U5[1] = 1; U5[4] = 1
    V5 = [0]*9; V5[4] = 1
    W5 = [0]*9; W5[0] = 1; W5[1] = 1
    strassen_terms.append((U5, V5, W5))
    
    # Term 6: M6 = (A10 + A20) * (B00 + B10)
    U6 = [0]*9; U6[3] = 1; U6[6] = 1
    V6 = [0]*9; V6[0] = 1; V6[3] = 1
    W6 = [0]*9; W6[6] = 1; W6[7] = 1; W6[8] = 1
    strassen_terms.append((U6, V6, W6))
    
    # Term 7: M7 = (A11 + A22) * (B11 + B21)
    U7 = [0]*9; U7[4] = 1; U7[8] = 1
    V7 = [0]*9; V7[4] = 1; V7[7] = 1
    W7 = [0]*9; W7[0] = 1; W7[4] = 1; W7[6] = 1
    strassen_terms.append((U7, V7, W7))
    
    # Now build the full decomposition for 3x3 * 3x6.
    # U has shape (9, 14), V has shape (18, 14), W has shape (18, 14)
    
    rank = 14
    U = [[0.0] * rank for _ in range(n * m)]  # (9, 14)
    V = [[0.0] * rank for _ in range(m * p)]  # (18, 14)
    W = [[0.0] * rank for _ in range(p * n)]  # (18, 14)
    
    # Mapping logic:
    # U index a = i*3 + j (0..8)
    # V index b = j*6 + k (0..17)
    # W index c = k*3 + i (0..17)
    
    for block_idx in range(2):
        for term_idx in range(7):
            U_t, V_t, W_t = strassen_terms[term_idx]
            col = block_idx * 7 + term_idx
            
            # U mapping: direct
            for a in range(9):
                U[a][col] = U_t[a]
            
            # V mapping:
            # V_t index b_3x3 = j*3 + k_prime (0..8)
            # j = b_3x3 // 3, k_prime = b_3x3 % 3
            # Block 0: k = k_prime. b_full = j*6 + k_prime
            # Block 1: k = k_prime + 3. b_full = j*6 + k_prime + 3
            for b_3x3 in range(9):
                val = V_t[b_3x3]
                if val == 0: continue
                j = b_3x3 // 3
                k_prime = b_3x3 % 3
                if block_idx == 0:
                    b_full = j * 6 + k_prime
                else:
                    b_full = j * 6 + k_prime + 3
                V[b_full][col] = val
            
            # W mapping:
            # W_t index c_3x3 = k_prime*3 + i (0..8)
            # k_prime = c_3x3 // 3, i = c_3x3 % 3
            # Block 0: k = k_prime. c_full = k_prime*3 + i
            # Block 1: k = k_prime + 3. c_full = (k_prime+3)*3 + i = 3*k_prime + 9 + i
            for c_3x3 in range(9):
                val = W_t[c_3x3]
                if val == 0: continue
                k_prime = c_3x3 // 3
                i = c_3x3 % 3
                if block_idx == 0:
                    c_full = k_prime * 3 + i
                else:
                    c_full = 3 * k_prime + 9 + i
                W[c_full][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
