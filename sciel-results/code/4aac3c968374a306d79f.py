import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 29  # 10 + 7 + 12
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: A1 (2x1) * B1 (1x5)
    # A1 corresponds to A[:, 0] (j=0)
    # B1 corresponds to B[0, :] (j=0)
    # C contribution to all C[i, k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    # Part 2: A2 (2x2) * B2a (2x2) using Strassen
    # A2 corresponds to A[:, 1:3] (j=1, 2)
    # B2a corresponds to B[1:3, 0:2] (j=1, 2; k=0, 1)
    
    # Define global indices for A2 elements
    # a = A[0, 1] -> U[0*3 + 1] = U[1]
    # b = A[0, 2] -> U[0*3 + 2] = U[2]
    # c = A[1, 1] -> U[1*3 + 1] = U[4]
    # d = A[1, 2] -> U[1*3 + 2] = U[5]
    
    # Define global indices for B2a elements
    # e = B[1, 0] -> V[1*5 + 0] = V[5]
    # f = B[1, 1] -> V[1*5 + 1] = V[6]
    # g = B[2, 0] -> V[2*5 + 0] = V[10]
    # h = B[2, 1] -> V[2*5 + 1] = V[11]
    
    # Define global indices for W contributions
    # C[0, 0] -> W[0*2 + 0] = W[0]
    # C[0, 1] -> W[1*2 + 0] = W[2]
    # C[1, 0] -> W[0*2 + 1] = W[1]
    # C[1, 1] -> W[1*2 + 1] = W[3]
    
    strassen_steps = [
        # (U_coeffs dict, V_coeffs dict, W_coeffs dict)
        ({1: 1, 5: 1}, {5: 1, 11: 1}, {0: 1, 3: 1}),       # M1 = (a+d)(e+h)
        ({4: 1, 5: 1}, {5: 1}, {1: 1, 3: -1}),              # M2 = (c+d)e
        ({1: 1}, {6: 1, 11: -1}, {2: 1, 3: 1}),             # M3 = a(f-h)
        ({5: 1}, {10: 1, 5: -1}, {0: 1, 1: 1}),             # M4 = d(g-e)
        ({1: 1, 2: 1}, {11: 1}, {0: -1, 2: 1}),             # M5 = (a+b)h
        ({4: 1, 1: -1}, {5: 1, 6: 1}, {3: 1}),              # M6 = (c-a)(e+f)
        ({2: 1, 5: -1}, {10: 1, 11: 1}, {0: 1})             # M7 = (b-d)(g+h)
    ]
    
    for u_coeffs, v_coeffs, w_coeffs in strassen_steps:
        for idx, val in u_coeffs.items():
            U[idx, col] = val
        for idx, val in v_coeffs.items():
            V[idx, col] = val
        for idx, val in w_coeffs.items():
            W[idx, col] = val
        col += 1
        
    # Part 3: A2 (2x2) * B2b (2x3)
    # A2 corresponds to A[:, 1:3] (j=1, 2)
    # B2b corresponds to B[1:3, 2:5] (j=1, 2; k=2, 3, 4)
    
    for i in range(n):
        for j in [1, 2]:
            for k in [2, 3, 4]:
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
