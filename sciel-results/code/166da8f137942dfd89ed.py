import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12
    
    # Target rank: 33
    # 3 blocks of 2x2 Strassen (7 each) + 12 schoolbook terms for the remaining part
    target_rank = 33
    
    U = np.zeros((num_A, target_rank))
    V = np.zeros((num_B, target_rank))
    W = np.zeros((num_C, target_rank))
    
    col = 0
    
    # Part 1: A_1 * B_1 using Strassen
    # A_1 is rows 0-1, cols 0-1 of A (indices 0,1,3,4)
    # B_1 is rows 0-1 of B (indices 0-11), split into 3 blocks of 2 cols
    
    for t in range(3):
        # B block t uses columns 2t and 2t+1
        # B indices:
        # n_00 -> B[0, 2t] -> idx 2t
        # n_01 -> B[0, 2t+1] -> idx 2t+1
        # n_10 -> B[1, 2t] -> idx 6+2t
        # n_11 -> B[1, 2t+1] -> idx 6+2t+1
        
        b_n00 = 2*t
        b_n01 = 2*t + 1
        b_n10 = 6 + 2*t
        b_n11 = 6 + 2*t + 1
        
        # C indices for block t (cols 2t, 2t+1)
        # c_00 -> C[0, 2t] -> idx 4t
        # c_01 -> C[0, 2t+1] -> idx 4t+1
        # c_10 -> C[1, 2t] -> idx 4t+2
        # c_11 -> C[1, 2t+1] -> idx 4t+3
        
        c_p00 = 4*t
        c_p01 = 4*t + 1
        c_p10 = 4*t + 2
        c_p11 = 4*t + 3
        
        # A indices
        # a_m00 -> A[0,0] -> idx 0
        # a_m01 -> A[0,1] -> idx 1
        # a_m10 -> A[1,0] -> idx 3
        # a_m11 -> A[1,1] -> idx 4
        
        a_m00 = 0
        a_m01 = 1
        a_m10 = 3
        a_m11 = 4
        
        # Strassen Term 1: (m00 + m11) * (n00 + n11)
        # U: a_m00, a_m11
        # V: b_n00, b_n11
        # W: c_p00 (+1), c_p11 (+1)
        U[a_m00, col] += 1
        U[a_m11, col] += 1
        V[b_n00, col] += 1
        V[b_n11, col] += 1
        W[c_p00, col] += 1
        W[c_p11, col] += 1
        col += 1
        
        # Strassen Term 2: (m10 + m11) * n00
        # U: a_m10, a_m11
        # V: b_n00
        # W: c_p10 (+1), c_p11 (-1)
        U[a_m10, col] += 1
        U[a_m11, col] += 1
        V[b_n00, col] += 1
        W[c_p10, col] += 1
        W[c_p11, col] -= 1
        col += 1
        
        # Strassen Term 3: m00 * (n10 - n11)
        # U: a_m00
        # V: b_n10, b_n11
        # W: c_p01 (+1), c_p11 (+1)
        U[a_m00, col] += 1
        V[b_n10, col] += 1
        V[b_n11, col] -= 1
        W[c_p01, col] += 1
        W[c_p11, col] += 1
        col += 1
        
        # Strassen Term 4: m11 * (n01 - n00)
        # U: a_m11
        # V: b_n01, b_n00
        # W: c_p00 (+1), c_p10 (+1)
        U[a_m11, col] += 1
        V[b_n01, col] += 1
        V[b_n00, col] -= 1
        W[c_p00, col] += 1
        W[c_p10, col] += 1
        col += 1
        
        # Strassen Term 5: (m00 + m01) * n11
        # U: a_m00, a_m01
        # V: b_n11
        # W: c_p00 (-1), c_p01 (+1)
        U[a_m00, col] += 1
        U[a_m01, col] += 1
        V[b_n11, col] += 1
        W[c_p00, col] -= 1
        W[c_p01, col] += 1
        col += 1
        
        # Strassen Term 6: (m10 - m00) * (n00 + n01)
        # U: a_m10, a_m00
        # V: b_n00, b_n01
        # W: c_p11 (+1)
        U[a_m10, col] += 1
        U[a_m00, col] -= 1
        V[b_n00, col] += 1
        V[b_n01, col] += 1
        W[c_p11, col] += 1
        col += 1
        
        # Strassen Term 7: (m01 - m11) * (n10 + n11)
        # U: a_m01, a_m11
        # V: b_n10, b_n11
        # W: c_p00 (+1)
        U[a_m01, col] += 1
        U[a_m11, col] -= 1
        V[b_n10, col] += 1
        V[b_n11, col] += 1
        W[c_p00, col] += 1
        col += 1
        
    # Part 2: A_2 * B_2 using Schoolbook
    # A_2 is col 2 of A (indices 2, 5)
    # B_2 is row 2 of B (indices 12-17)
    
    for i in range(n):
        a_idx = i * m + 2  # 2 or 5
        for k in range(p):
            b_idx = 2 * p + k  # 12 to 17
            c_idx = k * n + i  # 2k + i
            
            U[a_idx, col] += 1
            V[b_idx, col] += 1
            W[c_idx, col] += 1
            col += 1
            
    # col should be 33
    if col != target_rank:
        # Fallback to schoolbook if something went wrong (should not happen)
        return _schoolbook_construct()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_construct():
    n, m, p = DIMENSIONS
    num_A = n * m
    num_B = m * p
    num_C = p * n
    rank = n * m * p
    U = np.zeros((num_A, rank))
    V = np.zeros((num_B, rank))
    W = np.zeros((num_C, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
