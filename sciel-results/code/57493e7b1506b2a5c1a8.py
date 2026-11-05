import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 150.0  # wall-clock limit

def _elapsed():
    return time.time() - _start_time

def _deadline_exceeded(margin=5.0):
    return _elapsed() > _DEADLINE - margin

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm (rank 7).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    U[i*2+j, k] corresponds to A[i, j]
    V[j*2+k, k] corresponds to B[j, k]
    W[k*2+i, k] corresponds to C[i, k]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen terms:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # Term 1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # Term 3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # Term 4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1
    
    # Term 5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = 1
    
    # Term 6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _rank11_2x2x3():
    """
    Returns U, V, W for 2x2x3 matrix multiplication using a rank 11 decomposition.
    Combines Strassen (2x2x2) and Schoolbook (2x2x1).
    U: (4, 11), V: (6, 11), W: (6, 11)
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Schoolbook for 2x2x1 (column 3 of B, which is indices 2 and 3 in V? No, V is j*p+k. p=3.
    # B is 2x3. Indices:
    # 0: (0,0), 1: (0,1), 2: (0,2)
    # 3: (1,0), 4: (1,1), 5: (1,2)
    # Strassen used B columns 0 and 1 (indices 0,1 and 3,4).
    # We need to handle column 2 (indices 2 and 5).
    
    rank = 11
    U = np.zeros((4, rank))
    V = np.zeros((6, rank))
    W = np.zeros((6, rank))
    
    # Copy Strassen part (first 7 columns)
    # But Strassen W was for 2x2 output. Here output is 2x3.
    # W indices for 2x3:
    # 0: (0,0), 1: (0,1), 2: (0,2)
    # 3: (1,0), 4: (1,1), 5: (1,2)
    # Strassen W calculated C[0,0], C[0,1], C[1,0], C[1,1].
    # So W_s[0] -> W[0], W_s[1] -> W[1], W_s[2] -> W[3], W_s[3] -> W[4].
    
    U[:, :7] = U_s
    V[:, :4] = V_s[:, :4] # V_s is 4x7. We only used cols 0,1 of B (indices 0,1 and 3,4).
    # Wait, V_s in _strassen_2x2x2 is 4x7.
    # V_s[0] is b00, V_s[1] is b01, V_s[2] is b10, V_s[3] is b11.
    # In 2x2x3, V is 6x11.
    # V[0]=b00, V[1]=b01, V[2]=b02
    # V[3]=b10, V[4]=b11, V[5]=b12
    
    # Map Strassen V to 2x2x3 V
    V[0, :7] = V_s[0, :]
    V[1, :7] = V_s[1, :]
    V[3, :7] = V_s[2, :]
    V[4, :7] = V_s[3, :]
    
    # Map Strassen W to 2x2x3 W
    W[0, :7] = W_s[0, :]
    W[1, :7] = W_s[1, :]
    W[3, :7] = W_s[2, :]
    W[4, :7] = W_s[3, :]
    
    # Schoolbook for column 2 (indices 2 and 5 in V)
    # C[0,2] = a00*b02 + a01*b12
    # C[1,2] = a10*b02 + a11*b12
    
    # Term 8: a00 * b02
    U[0, 7] = 1
    V[2, 7] = 1
    W[2, 7] = 1
    
    # Term 9: a01 * b12
    U[1, 8] = 1
    V[5, 8] = 1
    W[2, 8] = 1
    
    # Term 10: a10 * b02
    U[2, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1
    
    # Term 11: a11 * b12
    U[3, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Partition A (3x4) into A1 (2x4) and A2 (1x4)
    # Partition B (4x5) into B1 (4x3) and B2 (4x2)
    # C = A1*B1 + A1*B2 + A2*B1 + A2*B2
    
    # Sub-problems:
    # 1. A1 (2x4) * B1 (4x3) -> 2x3
    #    Split A1 into A11 (2x2), A12 (2x2)
    #    Split B1 into B11 (2x3), B12 (2x3)
    #    A1*B1 = A11*B11 + A12*B12
    #    Each is 2x2x3. Rank 11.
    #    Total rank: 11 + 11 = 22.
    
    # 2. A1 (2x4) * B2 (4x2) -> 2x2
    #    Split A1 into A11 (2x2), A12 (2x2)
    #    Split B2 into B21 (2x2), B22 (2x2)
    #    A1*B2 = A11*B21 + A12*B22
    #    Each is 2x2x2. Rank 7.
    #    Total rank: 7 + 7 = 14.
    
    # 3. A2 (1x4) * B1 (4x3) -> 1x3
    #    Rank 1x4x3 = 12.
    
    # 4. A2 (1x4) * B2 (4x2) -> 1x2
    #    Rank 1x4x2 = 8.
    
    total_rank = 22 + 14 + 12 + 8
    # 56
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # 1. A11 * B11 (2x2x3)
    U1, V1, W1 = _rank11_2x2x3()
    # Map to global indices
    # A11 is A[0:2, 0:2] -> U indices 0:2, 2:4 (i*m+j)
    # B11 is B[0:2, 0:3] -> V indices 0:3, 6:9 (j*p+k)
    # C11 is C[0:2, 0:3] -> W indices 0:3, 3:6 (k*n+i)
    
    for r in range(11):
        for i in range(2):
            for j in range(2):
                u_idx = i * m + j
                U[u_idx, col_offset + r] = U1[i * 2 + j, r]
        
        for j in range(2):
            for k in range(3):
                v_idx = j * p + k
                V[v_idx, col_offset + r] = V1[j * 3 + k, r]
        
        for k in range(3):
            for i in range(2):
                w_idx = k * n + i
                W[w_idx, col_offset + r] = W1[k * 2 + i, r]
                
    col_offset += 11
    
    # 2. A12 * B12 (2x2x3)
    U1, V1, W1 = _rank11_2x2x3()
    # A12 is A[0:2, 2:4] -> U indices 2:4, 4:6
    # B12 is B[2:4, 0:3] -> V indices 6:9, 12:15
    # C12 is C[0:2, 0:3] -> W indices 0:3, 3:6
    
    for r in range(11):
        for i in range(2):
            for j in range(2):
                u_idx = i * m + (j + 2)
                U[u_idx, col_offset + r] = U1[i * 2 + j, r]
        
        for j in range(2):
            for k in range(3):
                v_idx = (j + 2) * p + k
                V[v_idx, col_offset + r] = V1[j * 3 + k, r]
        
        for k in range(3):
            for i in range(2):
                w_idx = k * n + i
                W[w_idx, col_offset + r] = W1[k * 2 + i, r]
                
    col_offset += 11
    
    # 3. A11 * B21 (2x2x2)
    U1, V1, W1 = _strassen_2x2x2()
    # A11 is A[0:2, 0:2]
    # B21 is B[0:2, 3:5]
    # C21 is C[0:2, 3:5]
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                u_idx = i * m + j
                U[u_idx, col_offset + r] = U1[i * 2 + j, r]
        
        for j in range(2):
            for k in range(2):
                v_idx = j * p + (k + 3)
                V[v_idx, col_offset + r] = V1[j * 2 + k, r]
        
        for k in range(2):
            for i in range(2):
                w_idx = (k + 3) * n + i
                W[w_idx, col_offset + r] = W1[k * 2 + i, r]
                
    col_offset += 7
    
    # 4. A12 * B22 (2x2x2)
    U1, V1, W1 = _strassen_2x2x2()
    # A12 is A[0:2, 2:4]
    # B22 is B[2:4, 3:5]
    # C22 is C[0:2, 3:5]
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                u_idx = i * m + (j + 2)
                U[u_idx, col_offset + r] = U1[i * 2 + j, r]
        
        for j in range(2):
            for k in range(2):
                v_idx = (j + 2) * p + (k + 3)
                V[v_idx, col_offset + r] = V1[j * 2 + k, r]
        
        for k in range(2):
            for i in range(2):
                w_idx = (k + 3) * n + i
                W[w_idx, col_offset + r] = W1[k * 2 + i, r]
                
    col_offset += 7
    
    # 5. A2 * B1 (1x4x3)
    # A2 is A[2, 0:4]
    # B1 is B[0:4, 0:3]
    # C21 is C[2, 0:3]
    # Rank 12
    
    for j in range(4):
        for k in range(3):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            
            U[u_idx, col_offset] = 1.0
            V[v_idx, col_offset] = 1.0
            W[w_idx, col_offset] = 1.0
            col_offset += 1
            
    col_offset += 0 # Already incremented in loop? No, I used col_offset directly.
    # Fix: The loop above used col_offset without incrementing it properly for each term.
    # Let's rewrite the schoolbook part more carefully.
    
    # Reset col_offset to start of schoolbook part
    # Actually, I just used col_offset in the loop. I need to increment it.
    # Let's restart the schoolbook part logic.
    
    # The previous loop:
    # for j... for k... U[...] = 1... col_offset += 1
    # This is correct.
    
    # 6. A2 * B2 (1x4x2)
    # A2 is A[2, 0:4]
    # B2 is B[0:4, 3:5]
    # C22 is C[2, 3:5]
    # Rank 8
    
    for j in range(4):
        for k in range(2):
            u_idx = 2 * m + j
            v_idx = j * p + (k + 3)
            w_idx = (k + 3) * n + 2
            
            U[u_idx, col_offset] = 1.0
            V[v_idx, col_offset] = 1.0
            W[w_idx, col_offset] = 1.0
            col_offset += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
