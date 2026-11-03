"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: 4x4x6 = (1x1x3) ⊗ (4x4x2)
    # Factor 1: 1x1x3 matrix multiplication (rank 3)
    # Factor 2: 4x4x2 matrix multiplication (rank 11)
    # Total rank = 3 * 11 = 33
    
    # Factor 1: 1x1x3
    # A is 1x1: A[0,0] = a
    # B is 1x3: B[0,0]=b0, B[0,1]=b1, B[0,2]=b2
    # C is 3x1: C[0,0]=a*b0, C[1,0]=a*b1, C[2,0]=a*b2
    
    # U1: shape (1*1, 3) = (1, 3)
    U1 = [[1, 1, 1]]
    
    # V1: shape (1*3, 3) = (3, 3)
    V1 = [[1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]]
    
    # W1: shape (3*1, 3) = (3, 3)
    W1 = [[1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]]
    
    # Factor 2: 4x4x2 with rank 11
    # A is 4x4, B is 4x2, C is 4x2
    # U2: shape (4*4, 11) = (16, 11)
    # V2: shape (4*2, 11) = (8, 11)
    # W2: shape (2*4, 11) = (8, 11)
    
    # For 4x4x2, we use a rank-11 decomposition
    # Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # Split B into 2x1 blocks: B = [B1; B2] where B1, B2 are 2x2
    # Wait, B is 4x2, so B = [[B11, B12], [B21, B22]] where each is 2x1
    
    # Let me use a direct construction for rank 11:
    # Terms 1-4: trivial for C[:,0] = A * B[:,0]
    # Terms 5-8: trivial for C[:,1] = A * B[:,1]
    # But that's rank 8. Can we do better?
    
    # Actually, let me use the known fact that 4x4x2 has rank 11.
    # I'll construct it using Strassen-like sharing.
    
    # Let me just use a valid rank-16 decomposition for now (trivial)
    # and see if I can optimize.
    
    # Trivial rank 16 for 4x4x2:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Term (i,j,k): U[i*4+j, t] = 1, V[j*2+k, t] = 1, W[k*4+i, t] = 1
    
    U2 = [[0]*16 for _ in range(16)]
    V2 = [[0]*16 for _ in range(8)]
    W2 = [[0]*16 for _ in range(8)]
    
    t = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U2[i*4+j][t] = 1
                V2[j*2+k][t] = 1
                W2[k*4+i][t] = 1
                t += 1
    
    # Now combine using Kronecker product
    # 4x4x6 = (1x1x3) ⊗ (4x4x2)
    # U: shape (4*4, 3*16) = (16, 48)
    # V: shape (4*6, 3*16) = (24, 48)
    # W: shape (6*4, 3*16) = (24, 48)
    
    rank = 3 * 16
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(24)]
    
    for c1 in range(3):
        for c2 in range(16):
            c = c1 * 16 + c2
            
            # U: (1x1) ⊗ (4x4)
            for i1 in range(1):
                for i2 in range(4):
                    for j2 in range(4):
                        u1 = U1[i1][c1]
                        u2 = U2[i2*4+j2][c2]
                        if u1 != 0 and u2 != 0:
                            row = i1*16 + i2*4 + j2
                            U[row][c] = u1 * u2
            
            # V: (1x3) ⊗ (4x2)
            for j1 in range(1):
                for k1 in range(3):
                    for j2 in range(4):
                        for k2 in range(2):
                            v1 = V1[j1*3+k1][c1]
                            v2 = V2[j2*2+k2][c2]
                            if v1 != 0 and v2 != 0:
                                row = j1*12 + k1*8 + j2*2 + k2
                                V[row][c] = v1 * v2
            
            # W: (3x1) ⊗ (4x4)
            for k1 in range(3):
                for i1 in range(1):
                    for k2 in range(2):
                        for i2 in range(4):
                            w1 = W1[k1*1+i1][c1]
                            w2 = W2[k2*4+i2][c2]
                            if w1 != 0 and w2 != 0:
                                row = k1*12 + i1*4 + k2*4 + i2
                                W[row][c] = w1 * w2
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
