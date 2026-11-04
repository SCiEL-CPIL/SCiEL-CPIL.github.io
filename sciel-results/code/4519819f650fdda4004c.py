import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 multiplication using Strassen (Rank 49)
    # Strassen for 2x2x2
    # U1, V1, W1 are 4x7
    U_222 = np.zeros((4, 7))
    V_222 = np.zeros((4, 7))
    W_222 = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # A11=0,0; A22=1,1 -> U[0]=1, U[3]=1
    # B11=0,0; B22=1,1 -> V[0]=1, V[3]=1
    # C11 += M1, C22 += M1 -> W[0]=1, W[3]=1
    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1
    
    # M2 = (A21+A22)B11
    # A21=1,0; A22=1,1 -> U[2]=1, U[3]=1
    # B11=0,0 -> V[0]=1
    # C21 += M2 -> W[2]=1
    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[2, 1] = 1
    
    # M3 = A11(B12-B22)
    # A11=0,0 -> U[0]=1
    # B12=0,1; B22=1,1 -> V[1]=1, V[3]=-1
    # C12 += M3 -> W[1]=1
    U_222[0, 2] = 1
    V_222[1, 2] = 1; V_222[3, 2] = -1
    W_222[1, 2] = 1
    
    # M4 = A22(B21-B11)
    # A22=1,1 -> U[3]=1
    # B21=1,0; B11=0,0 -> V[2]=1, V[0]=-1
    # C11 += M4, C21 += M4 -> W[0]=1, W[2]=1
    U_222[3, 3] = 1
    V_222[2, 3] = 1; V_222[0, 3] = -1
    W_222[0, 3] = 1; W_222[2, 3] = 1
    
    # M5 = (A11+A12)B22
    # A11=0,0; A12=0,1 -> U[0]=1, U[1]=1
    # B22=1,1 -> V[3]=1
    # C11 -= M5, C12 += M5 -> W[0]=-1, W[1]=1
    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = -1; W_222[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # A21=1,0; A11=0,0 -> U[2]=1, U[0]=-1
    # B11=0,0; B12=0,1 -> V[0]=1, V[1]=1
    # C22 += M6 -> W[3]=1
    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    W_222[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A12=0,1; A22=1,1 -> U[1]=1, U[3]=-1
    # B21=1,0; B22=1,1 -> V[2]=1, V[3]=1
    # C11 += M7 -> W[0]=1
    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    W_222[0, 6] = 1

    # Kronecker product for 4x4x4
    # U_444 = kron(U_222, U_222)
    U_444 = np.kron(U_222, U_222)
    V_444 = np.kron(V_222, V_222)
    W_444 = np.kron(W_222, W_222)
    
    # Map to target indices for Part 1 (k=0..3)
    U_part1 = U_444 # 16x49
    V_part1 = np.zeros((28, 49))
    W_part1 = np.zeros((28, 49))
    
    for j in range(4):
        for kp in range(4): # kp is local k (0..3)
            V_part1[7*j + kp, :] = V_444[4*j + kp, :]
            W_part1[4*kp + i, :] = W_444[4*kp + i, :] # Wait, W index mapping needs i loop
    
    # Correct W mapping
    W_part1 = np.zeros((28, 49))
    for kp in range(4):
        for i in range(4):
            W_part1[4*kp + i, :] = W_444[4*kp + i, :]

    # Part 2: 4x4x3 multiplication using Schoolbook (Rank 48)
    U_part2 = np.zeros((16, 48))
    V_part2 = np.zeros((28, 48))
    W_part2 = np.zeros((28, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for kp in range(3): # kp is local k (0..2), global k = kp + 4
                U_part2[4*i + j, col] = 1
                V_part2[7*j + (kp + 4), col] = 1
                W_part2[4*(kp + 4) + i, col] = 1
                col += 1

    # Combine
    U = np.hstack([U_part1, U_part2])
    V = np.hstack([V_part1, V_part2])
    W = np.hstack([W_part1, W_part2])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
