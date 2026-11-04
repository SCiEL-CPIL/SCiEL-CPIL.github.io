import numpy as np
import time

def construct():
    n, m, p = 2, 3, 4
    
    # Build the full tensor for verification
    N_U = n * m  # 6
    N_V = m * p  # 12
    N_W = p * n  # 8
    
    # Target tensor
    T = np.zeros((N_U, N_V, N_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Strategy: Split A(2x3) into A01(2x2) and A2(2x1) by columns j=0,1 and j=2
    # Split B(3x4) into B01(2x4) and B2(1x4) by rows j=0,1 and j=2
    # C = A01 * B01 + A2 * B2
    #
    # A01(2x2) * B01(2x4): Split B01 into B01a(2x2) and B01b(2x2) by columns k=0,1 and k=2,3
    #   A01 * B01a: 2x2 x 2x2, rank 7 (Strassen)
    #   A01 * B01b: 2x2 x 2x2, rank 7 (Strassen)
    #
    # A2(2x1) * B2(1x4): Split B2 into B2a(1x2) and B2b(1x2) by columns k=0,1 and k=2,3
    #   A2 * B2a: 2x1 x 1x2, rank 4
    #   A2 * B2b: 2x1 x 1x2, rank 4
    #
    # Total: 7 + 7 + 4 + 4 = 22
    
    def strassen_2x2():
        """Return U, V, W for 2x2 x 2x2 matrix multiplication, rank 7"""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B12+B21)
        U[0, 0] = 1; U[3, 0] = 1
        V[1, 0] = 1; V[2, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3 = A11(B11-B21)
        U[0, 2] = 1
        V[0, 2] = 1; V[2, 2] = -1
        W[0, 2] = 1; W[1, 2] = 1
        
        # M4 = A22(B22-B12)
        U[3, 3] = 1
        V[3, 3] = 1; V[1, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[1, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    # Build the rank-22 decomposition
    rank = 22
    U = np.zeros((N_U, rank))
    V = np.zeros((N_V, rank))
    W = np.zeros((N_W, rank))
    
    col = 0
    
    # Block 1: A01(2x2) * B01a(2x2) -> C block (2x2) for k=0,1
    # A indices: i*m+j for i in {0,1}, j in {0,1} -> 0,1,2,3
    # B indices: j*p+k for j in {0,1}, k in {0,1} -> 0,1,4,5
    # C indices: k*n+i for k in {0,1}, i in {0,1} -> 0,1,2,3
    
    sU, sV, sW = strassen_2x2()
    
    # Map A positions: (i,j) -> i*3+j for j=0,1
    # Strassen U has rows for (0,0),(0,1),(1,0),(1,1) -> our indices 0,1,2,3
    U[0, col:col+7] = sU[0]  # (0,0)
    U[1, col:col+7] = sU[1]  # (0,1)
    U[2, col:col+7] = sU[2]  # (1,0)
    U[3, col:col+7] = sU[3]  # (1,1)
    
    # Map B positions: (j,k) -> j*4+k for j=0,1, k=0,1
    # Strassen V has rows for (0,0),(0,1),(1,0),(1,1) -> our indices 0,1,4,5
    V[0, col:col+7] = sV[0]  # (0,0)
    V[1, col:col+7] = sV[1]  # (0,1)
    V[4, col:col+7] = sV[2]  # (1,0)
    V[5, col:col+7] = sV[3]  # (1,1)
    
    # Map C positions: (k,i) -> k*2+i for k=0,1, i=0,1
    # Strassen W has rows for (0,0),(0,1),(1,0),(1,1) -> our indices 0,1,2,3
    W[0, col:col+7] = sW[0]  # (0,0)
    W[1, col:col+7] = sW[1]  # (0,1)
    W[2, col:col+7] = sW[2]  # (1,0)
    W[3, col:col+7] = sW[3]  # (1,1)
    
    col += 7
    
    # Block 2: A01(2x2) * B01b(2x2) -> C block (2x2) for k=2,3
    # A indices: same as above 0,1,2,3
    # B indices: j*p+k for j=0,1, k=2,3 -> 2,3,6,7
    # C indices: k*n+i for k=2,3, i=0,1 -> 4,5,6,7
    
    sU, sV, sW = strassen_2x2()
    
    U[0, col:col+7] = sU[0]
    U[1, col:col+7] = sU[1]
    U[2, col:col+7] = sU[2]
    U[3, col:col+7] = sU[3]
    
    V[2, col:col+7] = sV[0]  # (0,2)
    V[3, col:col+7] = sV[1]  # (0,3)
    V[6, col:col+7] = sV[2]  # (1,2)
    V[7, col:col+7] = sV[3]  # (1,3)
    
    W[4, col:col+7] = sW[0]  # (2,0)
    W[5, col:col+7] = sW[1]  # (2,1)
    W[6, col:col+7] = sW[2]  # (3,0)
    W[7, col:col+7] = sW[3]  # (3,1)
    
    col += 7
    
    # Block 3: A2(2x1) * B2a(1x2) -> C block (2x2) for k=0,1
    # A indices: i*3+2 for i=0,1 -> 2,5
    # B indices: 2*4+k for k=0,1 -> 8,9
    # C indices: k*2+i for k=0,1, i=0,1 -> 0,1,2,3
    
    # 2x1 x 1x2: C[i,k] = A2[i] * B2a[k]
    # 4 rank-1 terms: (i,k) pairs
    for i in range(2):
        for k in range(2):
            a_idx = i * 3 + 2  # A2 row i
            b_idx = 2 * 4 + k  # B2a col k
            c_idx = k * 2 + i  # C position (k,i)
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
    
    # Block 4: A2(2x1) * B2b(1x2) -> C block (2x2) for k=2,3
    # A indices: i*3+2 for i=0,1 -> 2,5
    # B indices: 2*4+k for k=2,3 -> 10,11
    # C indices: k*2+i for k=2,3, i=0,1 -> 4,5,6,7
    
    for i in range(2):
        for k in range(2):
            a_idx = i * 3 + 2
            b_idx = 2 * 4 + (k + 2)  # k=2,3 -> 10,11
            c_idx = (k + 2) * 2 + i  # C position
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1
    
    # Verify
    error = 0
    for a in range(N_U):
        for b in range(N_V):
            for c in range(N_W):
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val - T[a, b, c]) > 1e-10:
                    error += abs(val - T[a, b, c])
    
    if error > 1e-10:
        raise ValueError(f"Decomposition failed verification, error={error}")
    
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
