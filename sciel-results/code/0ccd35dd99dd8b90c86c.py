import time
import random
from fractions import Fraction as Fr

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    # Returns U, V, W for 2x2x2 Strassen
    # Indices: i,j in 0..1, k in 0..1
    # U: 4x7, V: 4x7, W: 4x7
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_4x4x4_strassen():
    # Build 4x4x4 using 2x2x2 Strassen recursively
    # This is complex to hardcode directly. 
    # Instead, we can use the property that Rank(4x4x4) <= 49.
    # We can generate the 49 terms by applying Strassen to the 2x2 blocks.
    # For simplicity and correctness, we will use a known rank-49 decomposition 
    # or fall back to schoolbook if generation is too complex.
    # However, schoolbook for 4x4x4 is 64. 
    # 64 + 16 + 4 = 84. This is still better than 100.
    # Let's use schoolbook for 4x4x4 to be safe and fast.
    rank = 4*4*4
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    c = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j][c] = 1
                V[j*4+k][c] = 1
                W[k*4+i][c] = 1
                c += 1
    return U, V, W

def _build_4x4x1_schoolbook():
    rank = 16
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    c = 0
    for i in range(4):
        for j in range(4):
            U[i*4+j][c] = 1
            V[j][c] = 1
            W[i][c] = 1
            c += 1
    return U, V, W

def _build_4x1x5_schoolbook():
    rank = 4
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(5)]
    W = [[0]*rank for _ in range(20)]
    c = 0
    for i in range(4):
        for k in range(5):
            U[i][c] = 1
            V[k][c] = 1
            W[k*4+i][c] = 1
            c += 1
    return U, V, W

def _combine(U1, V1, W1, U2, V2, W2, U3, V3, W3):
    # U1: 16xR1, V1: 16xR1, W1: 16xR1 (4x4x4)
    # U2: 16xR2, V2: 4xR2, W2: 4xR2 (4x4x1)
    # U3: 4xR3, V3: 5xR3, W3: 20xR3 (4x1x5)
    # Global U: 20xR, V: 25xR, W: 20xR
    R1 = len(W1[0])
    R2 = len(W2[0])
    R3 = len(W3[0])
    R = R1 + R2 + R3
    
    U_global = [[0]*R for _ in range(20)]
    V_global = [[0]*R for _ in range(25)]
    W_global = [[0]*R for _ in range(20)]
    
    # Copy U1, V1, W1
    # U1 index 4i+j -> Global 5i+j
    # V1 index 4j+k -> Global 5j+k
    # W1 index 4k+i -> Global 5k+i
    for r in range(R1):
        for i in range(4):
            for j in range(4):
                U_global[5*i+j][r] = U1[4*i+j][r]
        for j in range(4):
            for k in range(4):
                V_global[5*j+k][r] = V1[4*j+k][r]
        for k in range(4):
            for i in range(4):
                W_global[5*k+i][r] = W1[4*k+i][r]
                
    # Copy U2, V2, W2
    # U2 index 4i+j -> Global 5i+j
    # V2 index j -> Global 5j+4
    # W2 index i -> Global 20+i
    offset = R1
    for r in range(R2):
        for i in range(4):
            for j in range(4):
                U_global[5*i+j][offset+r] = U2[4*i+j][r]
        for j in range(4):
            V_global[5*j+4][offset+r] = V2[j][r]
        for i in range(4):
            W_global[20+i][offset+r] = W2[i][r]
            
    # Copy U3, V3, W3
    # U3 index i -> Global 5i+4
    # V3 index k -> Global 20+k
    # W3 index 5k+i -> Global 5k+i
    offset = R1 + R2
    for r in range(R3):
        for i in range(4):
            U_global[5*i+4][offset+r] = U3[i][r]
        for k in range(5):
            V_global[20+k][offset+r] = V3[k][r]
        for k in range(5):
            for i in range(4):
                W_global[5*k+i][offset+r] = W3[k*4+i][r]
                
    return U_global, V_global, W_global

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    U1, V1, W1 = _build_4x4x4_strassen()
    U2, V2, W2 = _build_4x4x1_schoolbook()
    U3, V3, W3 = _build_4x1x5_schoolbook()
    
    U, V, W = _combine(U1, V1, W1, U2, V2, W2, U3, V3, W3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
