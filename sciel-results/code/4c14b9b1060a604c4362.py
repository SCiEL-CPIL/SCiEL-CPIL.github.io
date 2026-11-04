import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 part for first 4 columns
    # Strassen 2x2x2 vectors
    # Indices: 0:11, 1:12, 2:21, 3:22
    u_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Kronecker product for 4x4x4
    U_444 = []
    V_444 = []
    W_444 = []
    for k1 in range(7):
        for k2 in range(7):
            u_kron = np.kron(u_strassen[k1], u_strassen[k2])
            v_kron = np.kron(v_strassen[k1], v_strassen[k2])
            w_kron = np.kron(w_strassen[k1], w_strassen[k2])
            U_444.append(u_kron.tolist())
            V_444.append(v_kron.tolist())
            W_444.append(w_kron.tolist())
            
    # Total rank so far: 49
    # Now add schoolbook for the last 3 columns (cols 4,5,6 in 0-indexed B, which are cols 5,6,7 in 1-indexed)
    # B is 4x7. Cols 0,1,2,3 used by Strassen. Cols 4,5,6 used by schoolbook.
    U_school = []
    V_school = []
    W_school = []
    for i in range(n):
        for j in range(m):
            for k in range(3): # 3 columns
                u = [0] * (n * m)
                u[i * m + j] = 1
                U_school.append(u)
                
                v = [0] * (m * p)
                v[j * p + (4 + k)] = 1
                V_school.append(v)
                
                w = [0] * (p * n)
                # Output index: k*n + i? 
                # Convention: W[k*n+i, r]. 
                # Here k is the column index of C (0..6), i is row index (0..3).
                w[(4 + k) * n + i] = 1
                W_school.append(w)
                
    # Combine
    U = U_444 + U_school
    V = V_444 + V_school
    W = W_444 + W_school
    
    # Convert to lists of lists if not already
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    
    return {"U": U, "V": V, "W": W}
