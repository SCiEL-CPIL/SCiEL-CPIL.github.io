import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen 2x2 coefficients
    # u2[s, a, b], v2[s, c, d], w2[s, p, q]
    u2 = np.zeros((7, 2, 2))
    v2 = np.zeros((7, 2, 2))
    w2 = np.zeros((7, 2, 2))
    
    u2[0, 0, 0] = 1; u2[0, 1, 1] = 1
    v2[0, 0, 0] = 1; v2[0, 1, 1] = 1
    w2[0, 0, 0] = 1; w2[0, 1, 1] = 1
    
    u2[1, 1, 0] = 1; u2[1, 1, 1] = 1
    v2[1, 0, 0] = 1
    w2[1, 1, 0] = 1
    
    u2[2, 0, 0] = 1
    v2[2, 0, 1] = 1; v2[2, 1, 1] = -1
    w2[2, 0, 1] = 1
    
    u2[3, 1, 1] = 1
    v2[3, 1, 0] = 1; v2[3, 0, 0] = -1
    w2[3, 0, 0] = -1; w2[3, 1, 0] = 1
    
    u2[4, 0, 0] = 1; u2[4, 0, 1] = 1
    v2[4, 1, 1] = 1
    w2[4, 0, 1] = 1; w2[4, 1, 1] = 1
    
    u2[5, 1, 0] = 1; u2[5, 0, 0] = -1
    v2[5, 0, 0] = 1; v2[5, 0, 1] = 1
    w2[5, 0, 0] = 1; w2[5, 1, 0] = 1
    
    u2[6, 0, 1] = 1; u2[6, 1, 1] = -1
    v2[6, 1, 0] = 1; v2[6, 1, 1] = 1
    w2[6, 0, 0] = 1; w2[6, 0, 1] = 1
    
    # Outer Strassen terms
    # X_k and Y_k definitions
    # X_k[a,b] = sum_{i,j} Ax[k, a, b, i, j] A[i,j]
    # Y_k[c,d] = sum_{i,j} By[k, c, d, i, j] B[i,j]
    # C[i,j] = sum_{k,p,q} Cc[k, p, q, i, j] M_k[p,q]
    
    Ax = np.zeros((7, 2, 2, 4, 4))
    By = np.zeros((7, 2, 2, 4, 4))
    Cc = np.zeros((7, 2, 2, 4, 4))
    
    # M1: X = A00+A11, Y = B00+B11
    # A00: i<2, j<2. A11: i>=2, j>=2.
    for a in range(2):
        for b in range(2):
            for i in range(4):
                for j in range(4):
                    if (i < 2 and j < 2) or (i >= 2 and j >= 2):
                        Ax[0, a, b, i, j] = 1
                    if (i < 2 and j < 2) or (i >= 2 and j >= 2):
                        By[0, a, b, i, j] = 1
    # C00 = M1...
    Cc[0, 0, 0, 0, 0] = 1
    Cc[0, 0, 0, 1, 1] = 1
    Cc[0, 1, 1, 2, 2] = 1
    Cc[0, 1, 1, 3, 3] = 1
    
    # M2: X = A10+A11, Y = B00
    for a in range(2):
        for b in range(2):
            for i in range(4):
                for j in range(4):
                    if (i >= 2 and j < 2) or (i >= 2 and j >= 2):
                        Ax[1, a, b, i, j] = 1
                    if (i < 2 and j < 2):
                        By[1, a, b, i, j] = 1
    # C10 = M2...
    Cc[1, 0, 0, 2, 0] = 1
    Cc[1, 0, 1, 2, 1] = 1
    Cc[1, 1, 0, 3, 0] = 1
    Cc[1, 1, 1, 3, 1] = 1
    
    # M3: X = A00, Y = B01-B11
    for a in range(2):
        for b in range(2):
            for i in range(4):
                for j in range(4):
                    if (i < 2 and j < 2):
                        Ax[2, a, b, i, j] = 1
                    if (i < 2 and j >= 2):
                        By[2, a, b, i, j] = 1
                    if (i >= 2 and j >= 2):
                        By[2, a, b, i, j] = -1
    # C01 = M3...
    Cc[2, 0, 0, 0, 2] = 1
    Cc[2, 0, 1, 0, 3] = 1
    Cc[2, 1, 0, 1, 2] = 1
    Cc[2, 1, 1, 1, 3] = 1
    
    # M4: X = A11, Y = B10-B00
    for a in range(2):
        for b in range(2):
            for i in range(4):
                for j in range(4):
                    if (i >= 2 and j >= 2):
                        Ax[3, a, b, i, j] = 1
                    if (i >= 2 and j < 2):
                        By[3, a, b, i, j] = 1
                    if (i < 2 and j < 2):
                        By[3, a, b, i, j] = -1
    # C00 -= M4, C10 += M4
    Cc[3, 0, 0, 0, 0] = -1
    Cc[3, 0, 1, 1, 1] = -1
    Cc[3, 1, 0, 2, 0] = 1
    Cc[3, 1, 1, 3, 1] = 1
    
    # M5: X = A00+A01, Y = B11
    for a in range(2):
        for b in range(2):
            for i in range(4):
                for j in range(4):
                    if (i < 2):
                        Ax[4, a, b, i, j] = 1
                    if (i >= 2 and j >= 2):
                        By[4, a, b, i, j] = 1
    # C01 += M5, C11 += M5
    Cc[4, 0, 0, 0, 2] = 1
    Cc[4, 0, 1, 0, 3] = 1
    Cc[4, 1, 0, 1, 2] = 1
    Cc[4, 1, 1, 1, 3] = 1
    Cc[4, 0, 0, 2, 2] = 1
    Cc[4, 0, 1, 2, 3] = 1
    Cc[4, 1, 0, 3, 2] = 1
    Cc[4, 1, 1, 3, 3] = 1
    
    # M6: X = A10-A00, Y = B00+B01
    for a in range(2):
        for b in range(2):
            for i in range(4):
                for j in range(4):
                    if (i >= 2 and j < 2):
                        Ax[5, a, b, i, j] = 1
                    if (i < 2 and j < 2):
                        Ax[5, a, b, i, j] = -1
                    if (i < 2):
                        By[5, a, b, i, j] = 1
    # C00 += M6
    Cc[5, 0, 0, 0, 0] = 1
    Cc[5, 0, 1, 1, 1] = 1
    Cc[5, 1, 0, 2, 2] = 1
    Cc[5, 1, 1, 3, 3] = 1
    
    # M7: X = A01-A11, Y = B10+B11
    for a in range(2):
        for b in range(2):
            for i in range(4):
                for j in range(4):
                    if (i < 2 and j >= 2):
                        Ax[6, a, b, i, j] = 1
                    if (i >= 2 and j >= 2):
                        Ax[6, a, b, i, j] = -1
                    if (i >= 2):
                        By[6, a, b, i, j] = 1
    # C00 += M7, C11 += M7
    Cc[6, 0, 0, 0, 0] = 1
    Cc[6, 0, 1, 1, 1] = 1
    Cc[6, 1, 0, 2, 2] = 1
    Cc[6, 1, 1, 3, 3] = 1
    Cc[6, 0, 0, 2, 2] = 1
    Cc[6, 0, 1, 2, 3] = 1
    Cc[6, 1, 0, 3, 2] = 1
    Cc[6, 1, 1, 3, 3] = 1
    
    # Now fill U, V, W
    for k in range(7):
        for s in range(7):
            r = k * 7 + s
            for i in range(4):
                for j in range(4):
                    # U[(i,j), r]
                    u_val = 0
                    for a in range(2):
                        for b in range(2):
                            u_val += u2[s, a, b] * Ax[k, a, b, i, j]
                    U[i*4+j, r] = u_val
                    
                    # V[(i,j), r]
                    v_val = 0
                    for c in range(2):
                        for d in range(2):
                            v_val += v2[s, c, d] * By[k, c, d, i, j]
                    V[i*4+j, r] = v_val
                    
                    # W[(i,j), r]
                    w_val = 0
                    for p in range(2):
                        for q in range(2):
                            w_val += w2[s, p, q] * Cc[k, p, q, i, j]
                    W[i*4+j, r] = w_val

    return {"U": U, "V": V, "W": W}
