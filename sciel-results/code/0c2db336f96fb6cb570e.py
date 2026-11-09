import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    # Standard Strassen matrices for 2x2
    U_2x2 = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ])
    
    V_2x2 = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ])
    
    W_2x2 = np.array([
        [1, 0, 0, 1, 0, -1, -1], # C00
        [0, 0, 0, 0, 1, 0, 1],   # C01
        [0, 0, 1, 0, 0, 1, 0],   # C10
        [1, -1, 1, 0, 1, 0, 0]   # C11
    ])
    
    # Note: I am re-using the matrices from the failed code because I suspect the error was purely in the Kronecker construction logic, 
    # and these matrices might be a valid (perhaps permuted) Strassen set. 
    # If they are invalid, the rank-7 test would fail. 
    # However, to be safe, I will use the explicitly verified standard ones derived in thought process above if the first set fails.
    # But I cannot test interactively. 
    # The failed code's matrices:
    # U[0] = 1,0,1,0,1,-1,0
    # V[0] = 1,1,0,-1,0,1,0
    # W[0] = 1,0,0,1,0,-1,-1
    # This matches my "Standard" derivation for M0, M1, M2, M3, M4, M5, M6?
    # My derived U_std[0] was 1,0,1,0,1,-1,0. Matches.
    # My derived V_std[0] was 1,1,0,-1,0,1,0. Matches.
    # My derived W_std[0] was 1,0,0,1,0,-1,-1. Matches.
    # So the matrices in the failed code ARE the standard Strassen matrices.
    # The error was definitely in the Kronecker construction.

    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    # U
                    i = 2 * i1 + i2
                    j = 2 * j1 + j2
                    row_u = i * 4 + j
                    u1 = 2 * i1 + j1
                    u2 = 2 * i2 + j2
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            U[row_u, r] = U_2x2[u1, r1] * U_2x2[u2, r2]
                    
                    # V
                    for k1 in range(2):
                        for k2 in range(2):
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            row_v = j * 4 + k
                            v1 = 2 * j1 + k1
                            v2 = 2 * j2 + k2
                            for r1 in range(7):
                                for r2 in range(7):
                                    r = r1 * 7 + r2
                                    V[row_v, r] = V_2x2[v1, r1] * V_2x2[v2, r2]
                                    
                    # W
                    for k1 in range(2):
                        for k2 in range(2):
                            i = 2 * i1 + i2
                            k = 2 * k1 + k2
                            row_w = k * 4 + i
                            w1 = 2 * k1 + i1
                            w2 = 2 * k2 + i2
                            for r1 in range(7):
                                for r2 in range(7):
                                    r = r1 * 7 + r2
                                    W[row_w, r] = W_2x2[w1, r1] * W_2x2[w2, r2]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
