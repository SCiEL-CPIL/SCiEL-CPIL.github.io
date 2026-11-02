import numpy as np

def construct():
    n, m, p = 2, 2, 10
    nm, mp, pn = n * m, m * p, p * n  # 4, 20, 20
    
    # Strassen rank-7 terms for 2x2 matrix multiplication
    # U: A coefficients (4 entries)
    # V: B coefficients (4 entries)
    # W: C coefficients (4 entries)
    # Order: [C00, C10, C01, C11] in the vector, mapped to W indices
    # W index mapping:
    # c_vec[0] -> C[0, 2k] -> W[2k*n + 0]
    # c_vec[1] -> C[1, 2k] -> W[2k*n + 1]
    # c_vec[2] -> C[0, 2k+1] -> W[(2k+1)*n + 0]
    # c_vec[3] -> C[1, 2k+1] -> W[(2k+1)*n + 1]
    
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    rank = 35
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for k in range(p // 2):  # 5 blocks
        for a_vec, b_vec, c_vec in terms:
            for idx in range(nm):
                U[idx, col] = a_vec[idx]
            
            V[0 * p + 2 * k, col] = b_vec[0]
            V[0 * p + 2 * k + 1, col] = b_vec[1]
            V[1 * p + 2 * k, col] = b_vec[2]
            V[1 * p + 2 * k + 1, col] = b_vec[3]
            
            W[2 * k * n + 0, col] = c_vec[0]
            W[2 * k * n + 1, col] = c_vec[1]
            W[(2 * k + 1) * n + 0, col] = c_vec[2]
            W[(2 * k + 1) * n + 1, col] = c_vec[3]
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
