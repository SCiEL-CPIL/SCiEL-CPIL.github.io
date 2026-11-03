"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # We construct a decomposition of rank 81.
    # Part 1: M_4,4,4 using Strassen tensor product (Rank 49)
    # Part 2: M_4,4,2 using Schoolbook (Rank 32)
    
    # --- Part 1: M_4,4,4 ---
    # Strassen for 2x2 has 7 terms.
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # c=0
    U2[0,0] = 1; U2[3,0] = 1
    V2[0,0] = 1; V2[3,0] = 1
    W2[0,0] = 1; W2[3,0] = 1
    # c=1
    U2[2,1] = 1; U2[3,1] = 1
    V2[0,1] = 1
    W2[2,1] = 1; W2[3,1] = 1
    # c=2
    U2[0,2] = 1
    V2[1,2] = 1; V2[3,2] = -1
    W2[1,2] = 1
    # c=3
    U2[3,3] = 1
    V2[2,3] = 1; V2[0,3] = -1
    W2[2,3] = 1
    # c=4
    U2[0,4] = 1; U2[1,4] = 1
    V2[3,4] = 1
    W2[1,4] = 1; W2[3,4] = 1
    # c=5
    U2[2,5] = 1; U2[0,5] = -1
    V2[0,5] = 1; V2[1,5] = 1
    W2[2,5] = 1
    # c=6
    U2[1,6] = 1; U2[3,6] = -1
    V2[2,6] = 1; V2[3,6] = 1
    W2[2,6] = 1
    
    U4_4_4 = np.zeros((16, 49))
    V4_4_4 = np.zeros((16, 49))
    W4_4_4 = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U2[i1*2+j1, c1]
                    if u1 != 0:
                        for i0 in range(2):
                            for j0 in range(2):
                                u0 = U2[i0*2+j0, c2]
                                if u0 != 0:
                                    I = i1*2 + i0
                                    J = j1*2 + j0
                                    U4_4_4[I*4+J, c] = u1 * u0
                                
            for j1 in range(2):
                for k1 in range(2):
                    v1 = V2[j1*2+k1, c1]
                    if v1 != 0:
                        for j0 in range(2):
                            for k0 in range(2):
                                v0 = V2[j0*2+k0, c2]
                                if v0 != 0:
                                    J = j1*2 + j0
                                    K = k1*2 + k0
                                    V4_4_4[J*4+K, c] = v1 * v0
                                    
            for k1 in range(2):
                for i1 in range(2):
                    w1 = W2[k1*2+i1, c1]
                    if w1 != 0:
                        for k0 in range(2):
                            for i0 in range(2):
                                w0 = W2[k0*2+i0, c2]
                                if w0 != 0:
                                    K = k1*2 + k0
                                    I = i1*2 + i0
                                    W4_4_4[K*4+I, c] = w1 * w0
                                    
    # --- Part 2: M_4,4,2 ---
    # Schoolbook for 4x4 by 4x2: Rank 32.
    # U: 16 rows, V: 8 rows, W: 8 rows.
    U4_4_2 = np.zeros((16, 32))
    V4_4_2 = np.zeros((8, 32))
    W4_4_2 = np.zeros((8, 32))
    
    for i in range(4):
        for j in range(4):
            for k in range(2):
                c = i*8 + j*2 + k
                U4_4_2[i*4+j, c] = 1
                V4_4_2[j*2+k, c] = 1
                W4_4_2[k*4+i, c] = 1
                
    # Combine
    # U: 16 rows, 49+32=81 cols
    # V: 16+8=24 rows, 49+32=81 cols
    # W: 16+8=24 rows, 49+32=81 cols
    
    U = np.hstack([U4_4_4, U4_4_2])
    V = np.vstack([V4_4_4, V4_4_2])
    W = np.vstack([W4_4_4, W4_4_2])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
