import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s/150s kill
    
    n, m, p = 3, 3, 4
    rank = 34
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # --- Left Block (B columns 0, 1) ---
    # Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: 0(0,0), 1(0,1), 3(1,0), 4(1,1)
    # B indices: 0(0,0), 1(0,1), 4(1,0), 5(1,1)
    # C indices: 0(0,0), 3(0,1), 1(1,0), 4(1,1)
    
    # P1: (A11+A22)*(B11+B22)
    U[0, col] = 1; U[4, col] = 1
    V[0, col] = 1; V[5, col] = 1
    W[0, col] = 1; W[4, col] = 1
    col += 1
    
    # P2: (A21+A22)*B11
    U[3, col] = 1; U[4, col] = 1
    V[0, col] = 1
    W[1, col] = 1; W[4, col] = -1
    col += 1
    
    # P3: A11*(B21-B11)
    U[0, col] = 1
    V[4, col] = 1; V[0, col] = -1
    W[3, col] = 1; W[4, col] = 1
    col += 1
    
    # P4: A22*(B12-B22)
    U[4, col] = 1
    V[1, col] = 1; V[5, col] = -1
    W[0, col] = 1; W[1, col] = 1
    col += 1
    
    # P5: (A11+A12)*B22
    U[0, col] = 1; U[1, col] = 1
    V[5, col] = 1
    W[0, col] = -1; W[3, col] = 1
    col += 1
    
    # P6: (A21-A11)*(B11+B12)
    U[3, col] = 1; U[0, col] = -1
    V[0, col] = 1; V[1, col] = 1
    W[4, col] = 1
    col += 1
    
    # P7: (A12-A22)*(B21+B22)
    U[1, col] = 1; U[4, col] = -1
    V[4, col] = 1; V[5, col] = 1
    W[0, col] = 1
    col += 1
    
    # Direct terms for Left Block involving A[:,2] or A[2,:]
    # A[0,2] * B[2,0:2]
    U[2, col] = 1; V[8, col] = 1; W[0, col] = 1; col += 1 # C[0,0]
    U[2, col] = 1; V[9, col] = 1; W[3, col] = 1; col += 1 # C[0,1]
    
    # A[1,2] * B[2,0:2]
    U[5, col] = 1; V[8, col] = 1; W[1, col] = 1; col += 1 # C[1,0]
    U[5, col] = 1; V[9, col] = 1; W[4, col] = 1; col += 1 # C[1,1]
    
    # A[2,0:2] * B[0:2,0:2]
    U[6, col] = 1; V[0, col] = 1; W[2, col] = 1; col += 1 # C[2,0]
    U[6, col] = 1; V[1, col] = 1; W[5, col] = 1; col += 1 # C[2,1]
    U[7, col] = 1; V[4, col] = 1; W[2, col] = 1; col += 1 # C[2,0]
    U[7, col] = 1; V[5, col] = 1; W[5, col] = 1; col += 1 # C[2,1]
    
    # A[2,2] * B[2,0:2]
    U[8, col] = 1; V[8, col] = 1; W[2, col] = 1; col += 1 # C[2,0]
    U[8, col] = 1; V[9, col] = 1; W[5, col] = 1; col += 1 # C[2,1]
    
    # --- Right Block (B columns 2, 3) ---
    # Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    # A indices: 0(0,0), 1(0,1), 3(1,0), 4(1,1)
    # B indices: 2(0,2), 3(0,3), 6(1,2), 7(1,3)
    # C indices: 6(0,2), 9(0,3), 7(1,2), 10(1,3)
    
    # P1: (A11+A22)*(B11+B22)
    U[0, col] = 1; U[4, col] = 1
    V[2, col] = 1; V[7, col] = 1
    W[6, col] = 1; W[10, col] = 1
    col += 1
    
    # P2: (A21+A22)*B11
    U[3, col] = 1; U[4, col] = 1
    V[2, col] = 1
    W[7, col] = 1; W[10, col] = -1
    col += 1
    
    # P3: A11*(B21-B11)
    U[0, col] = 1
    V[6, col] = 1; V[2, col] = -1
    W[9, col] = 1; W[10, col] = 1
    col += 1
    
    # P4: A22*(B12-B22)
    U[4, col] = 1
    V[3, col] = 1; V[7, col] = -1
    W[6, col] = 1; W[7, col] = 1
    col += 1
    
    # P5: (A11+A12)*B22
    U[0, col] = 1; U[1, col] = 1
    V[7, col] = 1
    W[6, col] = -1; W[9, col] = 1
    col += 1
    
    # P6: (A21-A11)*(B11+B12)
    U[3, col] = 1; U[0, col] = -1
    V[2, col] = 1; V[3, col] = 1
    W[10, col] = 1
    col += 1
    
    # P7: (A12-A22)*(B21+B22)
    U[1, col] = 1; U[4, col] = -1
    V[6, col] = 1; V[7, col] = 1
    W[6, col] = 1
    col += 1
    
    # Direct terms for Right Block involving A[:,2] or A[2,:]
    # A[0,2] * B[2,2:4]
    U[2, col] = 1; V[10, col] = 1; W[6, col] = 1; col += 1 # C[0,2]
    U[2, col] = 1; V[11, col] = 1; W[9, col] = 1; col += 1 # C[0,3]
    
    # A[1,2] * B[2,2:4]
    U[5, col] = 1; V[10, col] = 1; W[7, col] = 1; col += 1 # C[1,2]
    U[5, col] = 1; V[11, col] = 1; W[10, col] = 1; col += 1 # C[1,3]
    
    # A[2,0:2] * B[0:2,2:4]
    U[6, col] = 1; V[2, col] = 1; W[8, col] = 1; col += 1 # C[2,2]
    U[6, col] = 1; V[3, col] = 1; W[11, col] = 1; col += 1 # C[2,3]
    U[7, col] = 1; V[6, col] = 1; W[8, col] = 1; col += 1 # C[2,2]
    U[7, col] = 1; V[7, col] = 1; W[11, col] = 1; col += 1 # C[2,3]
    
    # A[2,2] * B[2,2:4]
    U[8, col] = 1; V[10, col] = 1; W[8, col] = 1; col += 1 # C[2,2]
    U[8, col] = 1; V[11, col] = 1; W[11, col] = 1; col += 1 # C[2,3]
    
    # Verify correctness
    if time.time() < deadline:
        T_reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        T_expected = np.zeros((9, 12, 12), dtype=int)
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    T_expected[i*3+j, j*4+k, k*3+i] = 1
        
        if not np.array_equal(T_reconstructed, T_expected):
            # Fallback to trivial rank-36
            rank = 36
            U = np.zeros((9, rank), dtype=int)
            V = np.zeros((12, rank), dtype=int)
            W = np.zeros((12, rank), dtype=int)
            col = 0
            for i in range(3):
                for j in range(3):
                    for k in range(4):
                        U[i*3+j, col] = 1
                        V[j*4+k, col] = 1
                        W[k*3+i, col] = 1
                        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
