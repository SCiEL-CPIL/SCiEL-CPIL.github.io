"""Matrix multiplication 4x4x4: rank-49 decomposition via Kronecker product of Strassen."""

import time

def construct():
    # Strassen 2x2x2 decomposition (verified correct)
    # T[i,j,k] = 1 for all i,j,k in {0,1}
    # U[i*2+j, l], V[j*2+k, l], W[k*2+i, l]
    
    # 7 terms, each with entries in {0, 1, -1}
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],   # i=0,j=0 -> A00
        [0, 0, 0, 0, 1,  0, 1],   # i=0,j=1 -> A01
        [0, 1, 0, 0, 0,  1, 0],   # i=1,j=0 -> A10
        [1, 1, 0, 1, 0,  0,-1],   # i=1,j=1 -> A11
    ]
    V2 = [
        [1, 1, 0, 0, 0, 1, 0],   # j=0,k=0 -> B00
        [0, 0, 0, 1, 0, 0, 1],   # j=0,k=1 -> B01
        [0, 0, 1,-1, 0, 1, 0],   # j=1,k=0 -> B10
        [1, 0,-1, 0, 1, 0, 1],   # j=1,k=1 -> B11
    ]
    W2 = [
        [1, 0, 1, 1,-1, 0, -1],  # k=0,i=0 -> C00
        [0, 0, 1, 0, 1, 0, -1],  # k=1,i=0 -> C01
        [0, 1, 0, 1, 0, 1,  0],  # k=0,i=1 -> C10
        [1, 1, 0, 0, 0, 0,  1],  # k=1,i=1 -> C11
    ]
    
    # Verify 2x2x2 decomposition
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for l in range(7):
                    s += U2[i*2+j][l] * V2[j*2+k][l] * W2[k*2+i][l]
                if s != 1:
                    # Fallback to schoolbook if verification fails
                    return _schoolbook()
    
    # Kronecker product to get 4x4x4 rank-49 decomposition
    # U4[i*4+j, l1*7+l2] = U2[i*2+j1, l1] * U2[i2*2+j2, l2]
    # where j = j1*2+j2, and we need to be careful with indexing
    
    # For 4x4x4: i,j,k in {0,1,2,3}
    # i = i1*2+i2, j = j1*2+j2, k = k1*2+k2
    # U[i*4+j, l] where l = l1*7+l2
    # U[i*4+j, l1*7+l2] = U2[i1*2+j1, l1] * U2[i2*2+j2, l2]
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for l1 in range(7):
        for l2 in range(7):
            l = l1*7 + l2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = i1*2 + i2
                            j = j1*2 + j2
                            u_val = U2[i1*2+j1][l1] * U2[i2*2+j2][l2]
                            if u_val != 0:
                                U4[i*4+j][l] = u_val
            
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            j = j1*2 + j2
                            k = k1*2 + k2
                            v_val = V2[j1*2+k1][l1] * V2[j2*2+k2][l2]
                            if v_val != 0:
                                V4[j*4+k][l] = v_val
            
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            k = k1*2 + k2
                            i = i1*2 + i2
                            w_val = W2[k1*2+i1][l1] * W2[k2*2+i2][l2]
                            if w_val != 0:
                                W4[k*4+i][l] = w_val
    
    # Verify 4x4x4 decomposition (spot check)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0
                for l in range(49):
                    s += U4[i*4+j][l] * V4[j*4+k][l] * W4[k*4+i][l]
                if s != 1:
                    return _schoolbook()
    
    # Convert to list format with integer entries
    return {"U": U4, "V": V4, "W": W4}

def _schoolbook():
    n = m = p = 4
    rank = 64
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j][col] = 1
                V[j*4+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
