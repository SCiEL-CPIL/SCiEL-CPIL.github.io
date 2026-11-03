import numpy as np
import time

def strassen_2x2x2():
    U = np.array([[1,0,1,0,1,0,0],
                  [0,1,0,1,0,1,0],
                  [0,0,0,0,0,-1,1],
                  [1,1,0,0,0,0,-1]])
    V = np.array([[1,1,0,-1,0,1,0],
                  [0,0,1,0,0,1,0],
                  [0,0,0,1,0,0,1],
                  [1,0,-1,0,1,0,1]])
    W = np.array([[1,0,0,1,-1,0,1],
                  [0,0,1,0,1,0,0],
                  [0,1,0,1,0,0,0],
                  [1,0,0,0,0,1,1]])
    return U, V, W

def strassen_4x4x4():
    U2, V2, W2 = strassen_2x2x2()
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    return U4, V4, W4

def schoolbook_4x4x3():
    U = np.zeros((16, 48))
    V = np.zeros((12, 48))
    W = np.zeros((12, 48))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U[i*4+j][col] = 1
                V[j*3+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 150.0
    
    U1, V1, W1 = strassen_4x4x4()
    U2, V2, W2 = schoolbook_4x4x3()
    
    U = np.hstack([U1, U2])
    V = np.vstack([V1, V2])
    W = np.vstack([W1, W2])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
