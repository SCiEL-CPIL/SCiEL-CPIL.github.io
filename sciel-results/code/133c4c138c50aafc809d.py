"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 12
    mp = m * p  # 32
    pn = p * n  # 24
    
    start_time = time.time()
    deadline = start_time + 140
    
    # Strassen decomposition for 2x2x2 (rank 7)
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]], C = A*B
    # U2[i*2+j, c]: coeff of A[i,j] in factor c
    # V2[j*2+k, c]: coeff of B[j,k] in factor c
    # W2[k*2+i, c]: coeff of C[i,k] in factor c (W index = k*2+i)
    
    # C[0,0] = M1 - M2 + M3 + M6
    # C[1,0] = M2 + M4
    # C[0,1] = M3 + M5
    # C[1,1] = M1 + M4 - M5 + M7
    
    # W2[k*2+i]: W2[0]=C[0,0], W2[1]=C[1,0], W2[2]=C[0,1], W2[3]=C[1,1]
    
    U2 = np.array([
        [1, 0, 1, 0, 1, 1, 0],  # a00
        [0, 0, 0, 0, 1, 0, 1],  # a01
        [0, 1, 0, 0, 0, 1, 0],  # a10
        [1, 1, 0, 1, 0, 0, -1]  # a11
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, 1, 0, 1, 0],  # b00
        [0, 0, 0, 0, 0, 1, 0],  # b01
        [0, 0, 1, -1, 0, 0, 1], # b10
        [1, 0, -1, 0, 1, 0, 1]  # b11
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 1, 0, 0, 1, 0],   # C[0,0]
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0]
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1]
        [1, -1, 0, 1, -1, 0, 1]  # C[1,1]
    ], dtype=float)
    
    # Verify Strassen
    def verify_2x2(U2, V2, W2):
        A = np.array([[1,2],[3,4]], dtype=float)
        B = np.array([[5,6],[7,8]], dtype=float)
        C = np.zeros((2,2))
        for c in range(7):
            a_val = sum(U2[i*2+j, c] * A[i,j] for i in range(2) for j in range(2))
            b_val = sum(V2[j*2+k, c] * B[j,k] for j in range(2) for k in range(2))
            for i in range(2):
                for k in range(2):
                    C[i,k] += W2[k*2+i, c] * a_val * b_val
        return np.allclose(C, A @ B, atol=1e-9)
    
    strassen_ok = verify_2x2(U2, V2, W2)
    
    if not strassen_ok:
        # Fallback: schoolbook for 2x2x2
        U2 = np.zeros((4,4)); V2 = np.zeros((4,4)); W2 = np.zeros((4,4))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U2[i*2+j, col] = 1
                    V2[j*2+k, col] = 1
                    W2[k*2+i, col] = 1
                    col += 1
        rank2 = 4
    else:
        rank2 = 7
    
    # 3x2x4 tensor decomposition
    # n1=3, m1=2, p1=4
    # U1: (6, r1), V1: (8, r1), W1: (12, r1)
    # U1[i*2+j, c], V1[j*4+k, c], W1[k*3+i, c]
    
    # Schoolbook
    rank1 = 24
    U1 = np.zeros((6, rank1))
    V1 = np.zeros((8, rank1))
    W1 = np.zeros((12, rank1))
    col = 0
    for i in range(3):
        for j in range(2):
            for k in range(4):
                U1[i*2+j, col] = 1
                V1[j*4+k, col] = 1
                W1[k*3+i, col] = 1
                col += 1
    
    # Try to reduce rank
    def verify_3x2x4(U, V, W):
        n1, m1, p1 = 3, 2, 4
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    s = sum(U[i*m1+j, c] * V[j*p1+k, c] * W[k*n1+i, c] for c in range(U.shape[1]))
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check invalid
        valid_set = set()
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    valid_set.add((i*m1+j, j*p1+k, k*n1+i))
        random.seed(123)
        for _ in range(100):
            a = random.randint(0, 5)
            b = random.randint(0, 7)
            c = random.randint(0, 11)
            if (a,b,c) not in valid_set:
                s = sum(U[a, col] * V[b, col] * W[c, col] for col in range(U.shape[1]))
                if abs(s) > 1e-9:
                    return False
        return True
    
    # Greedy column removal
    for _ in range(50):
        if time.time() > deadline:
            break
        removed = False
        for col in range(U1.shape[1]):
            U_try = np.delete(U1, col, axis=1)
            V_try = np.delete(V1, col, axis=1)
            W_try = np.delete(W1, col, axis=1)
            if verify_3x2x4(U_try, V_try, W_try):
                U1 = U_try
                V1 = V_try
                W1 = W_try
                rank1 = U1.shape[1]
                removed = True
                break
        if not removed:
            break
    
    # Combine via Kronecker product
    rank_full = rank2 * rank1
    U_full = np.zeros((12, rank_full))
    V_full = np.zeros((32, rank_full))
    W_full = np.zeros((24, rank_full))
    
    for c2 in range(rank2):
        for c1 in range(rank1):
            c = c2 * rank1 + c1
            for i in range(3):
                for j1 in range(2):
                    for j2 in range(2):
                        j = j2 * 2 + j1
                        U_full[i*4 + j, c] = U2[j2*2+j1, c2] * U1[i*2+j1, c1]
            
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(4):
                        for k2 in range(2):
                            j = j2 * 2 + j1
                            k = k2 * 4 + k1
                            V_full[j*8 + k, c] = V2[j2*2+k2, c2] * V1[j1*4+k1, c1]
            
            for k1 in range(4):
                for k2 in range(2):
                    for i in range(3):
                        k = k2 * 4 + k1
                        W_full[k*3 + i, c] = W2[k2*2+i, c2] * W1[k1*3+i, c1]
    
    # Verify full decomposition
    def verify_full(U, V, W):
        n, m, p = 3, 4, 8
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                    if abs(s - 1.0) > 1e-9:
                        return False
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_set.add((i*m+j, j*p+k, k*n+i))
        random.seed(456)
        for _ in range(200):
            a = random.randint(0, 11)
            b = random.randint(0, 31)
            c = random.randint(0, 23)
            if (a,b,c) not in valid_set:
                s = sum(U[a, col] * V[b, col] * W[c, col] for col in range(U.shape[1]))
                if abs(s) > 1e-9:
                    return False
        return True
    
    if not verify_full(U_full, V_full, W_full):
        # Fallback to schoolbook
        rank_full = 96
        U_full = np.zeros((12, rank_full))
        V_full = np.zeros((32, rank_full))
        W_full = np.zeros((24, rank_full))
        col = 0
        for i in range(3):
            for j in range(4):
                for k in range(8):
                    U_full[i*4+j, col] = 1
                    V_full[j*8+k, col] = 1
                    W_full[k*3+i, col] = 1
                    col += 1
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END

