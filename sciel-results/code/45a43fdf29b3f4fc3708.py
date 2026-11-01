import time
import numpy as np

def construct():
    n, m, p = 2, 2, 15
    
    all_U, all_V, all_W = [], [], []
    
    # 6 Strassen blocks for columns (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
    for k0, k1 in [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11)]:
        terms = _strassen_terms(k0, k1)
        for (u_dict, v_dict, w_dict) in terms:
            u_entries = [(i*m+j, c) for (i,j), c in u_dict.items()]
            v_entries = [(j*p+k_idx, c) for (j,k_idx), c in v_dict.items()]
            w_entries = [(k_idx*n+i, c) for (k_idx,i), c in w_dict.items()]
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)
    
    # 7 Strassen terms for columns 12, 13
    k0, k1 = 12, 13
    terms = _strassen_terms(k0, k1)
    for (u_dict, v_dict, w_dict) in terms:
        u_entries = [(i*m+j, c) for (i,j), c in u_dict.items()]
        v_entries = [(j*p+k_idx, c) for (j,k_idx), c in v_dict.items()]
        w_entries = [(k_idx*n+i, c) for (k_idx,i), c in w_dict.items()]
        all_U.append(u_entries)
        all_V.append(v_entries)
        all_W.append(w_entries)
    
    # 3 additional terms for column 14
    # Term 1: A[0,0] * B[0,14] -> C[0,14]
    all_U.append([(0, 1)])  # U[0*2+0] = U[0]
    all_V.append([(0*p+14, 1)])  # V[0*2+14] = V[14]
    all_W.append([(14*n+0, 1)])  # W[14*2+0] = W[28]
    
    # Term 2: A[0,1] * B[1,14] -> C[0,14]
    all_U.append([(1, 1)])  # U[0*2+1] = U[1]
    all_V.append([(1*p+14, 1)])  # V[1*2+14] = V[16]
    all_W.append([(14*n+0, 1)])  # W[14*2+0] = W[28]
    
    # Term 3: (A[1,0]+A[1,1]) * (B[0,14]+B[1,14]) -> C[1,14]
    all_U.append([(2, 1), (3, 1)])  # U[1*2+0]=U[2], U[1*2+1]=U[3]
    all_V.append([(0*p+14, 1), (1*p+14, 1)])  # V[14], V[16]
    all_W.append([(14*n+1, 1)])  # W[14*2+1] = W[29]
    
    rank = len(all_U)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        for row, val in all_U[c]:
            U[row][c] = val
        for row, val in all_V[c]:
            V[row][c] = val
        for row, val in all_W[c]:
            W[row][c] = val
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}

def _strassen_terms(k0, k1):
    return [
        ({(0,0):1, (1,1):1}, {(0,k0):1, (1,k1):1}, {(k0,0):1, (k1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,k0):1}, {(k0,1):1, (k1,1):-1}),
        ({(0,0):1}, {(0,k1):1, (1,k1):-1}, {(k1,0):1, (k1,1):1}),
        ({(1,1):1}, {(1,k0):1, (0,k0):-1}, {(k0,0):1, (k0,1):1}),
        ({(0,0):1, (0,1):1}, {(1,k1):1}, {(k0,0):-1, (k1,0):1}),
        ({(1,0):1, (0,0):-1}, {(0,k0):1, (0,k1):1}, {(k1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,k0):1, (1,k1):1}, {(k0,0):1}),
    ]

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if val != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W
