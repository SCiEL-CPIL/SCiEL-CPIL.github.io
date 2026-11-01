import time
import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 25: 3 Strassen blocks (rank 7 each) + 1 Schoolbook block (rank 4)
    rank = 25
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2x2
    # U indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # V indices for block k0,k1: 
    #   j=0,k0=e, j=0,k1=f, j=1,k0=g, j=1,k1=h
    # W indices for block k0,k1:
    #   k0,i0=C00, k0,i1=C10, k1,i0=C01, k1,i1=C11
    
    # Strassen terms: (u, v, w)
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]

    col = 0
    
    # Process 3 Strassen blocks for k=0,1; 2,3; 4,5
    for block_idx in range(3):
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        
        for u_loc, v_loc, w_loc in strassen_terms:
            # Map U
            u_glob = np.zeros(n * m)
            u_glob[0] = u_loc[0] # a
            u_glob[1] = u_loc[1] # b
            u_glob[2] = u_loc[2] # c
            u_glob[3] = u_loc[3] # d
            
            # Map V
            v_glob = np.zeros(m * p)
            v_glob[0 * p + k0] = v_loc[0] # e
            v_glob[0 * p + k1] = v_loc[1] # f
            v_glob[1 * p + k0] = v_loc[2] # g
            v_glob[1 * p + k1] = v_loc[3] # h
            
            # Map W
            w_glob = np.zeros(p * n)
            w_glob[k0 * n + 0] = w_loc[0] # C00
            w_glob[k0 * n + 1] = w_loc[1] # C10
            w_glob[k1 * n + 0] = w_loc[2] # C01
            w_glob[k1 * n + 1] = w_loc[3] # C11
            
            U[:, col] = u_glob
            V[:, col] = v_glob
            W[:, col] = w_glob
            col += 1

    # Process 1 Schoolbook block for k=6
    k0 = 6
    
    # Term 1: a * B06
    u_t1 = np.zeros(4); u_t1[0] = 1
    v_t1 = np.zeros(14); v_t1[0 * p + 6] = 1
    w_t1 = np.zeros(14); w_t1[6 * n + 0] = 1
    U[:, col] = u_t1; V[:, col] = v_t1; W[:, col] = w_t1
    col += 1
    
    # Term 2: b * B16
    u_t2 = np.zeros(4); u_t2[1] = 1
    v_t2 = np.zeros(14); v_t2[1 * p + 6] = 1
    w_t2 = np.zeros(14); w_t2[6 * n + 0] = 1
    U[:, col] = u_t2; V[:, col] = v_t2; W[:, col] = w_t2
    col += 1
    
    # Term 3: c * B06
    u_t3 = np.zeros(4); u_t3[2] = 1
    v_t3 = np.zeros(14); v_t3[0 * p + 6] = 1
    w_t3 = np.zeros(14); w_t3[6 * n + 1] = 1
    U[:, col] = u_t3; V[:, col] = v_t3; W[:, col] = w_t3
    col += 1
    
    # Term 4: d * B16
    u_t4 = np.zeros(4); u_t4[3] = 1
    v_t4 = np.zeros(14); v_t4[1 * p + 6] = 1
    w_t4 = np.zeros(14); w_t4[6 * n + 1] = 1
    U[:, col] = u_t4; V[:, col] = v_t4; W[:, col] = w_t4
    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
