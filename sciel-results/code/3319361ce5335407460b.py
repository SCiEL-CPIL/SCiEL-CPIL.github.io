import numpy as np
import time

DIMENSIONS = [2, 4, 4]
DEADLINE = 140.0 # Safety margin from 150s

def _generate_strassen_decomposition():
    # Same as before, rank 28
    strassen_cols = [
        {0: [ (0,1), (3,1) ], 1: [ (0,1), (3,1) ], 2: [ (0,1), (3,1) ]},
        {0: [ (2,1), (3,1) ], 1: [ (0,1) ], 2: [ (2,1), (3,-1) ]},
        {0: [ (0,1) ], 1: [ (1,1), (3,-1) ], 2: [ (1,1), (3,1) ]},
        {0: [ (3,1) ], 1: [ (2,1), (0,-1) ], 2: [ (0,1), (2,1) ]},
        {0: [ (0,1), (1,1) ], 1: [ (3,1) ], 2: [ (0,-1), (1,1) ]},
        {0: [ (2,1), (0,-1) ], 1: [ (0,1), (1,1) ], 2: [ (3,1) ]},
        {0: [ (1,1), (3,-1) ], 1: [ (2,1), (3,1) ], 2: [ (0,1) ]}
    ]
    
    U = np.zeros((8, 28), dtype=int)
    V = np.zeros((16, 28), dtype=int)
    W = np.zeros((8, 28), dtype=int)
    
    def map_A(local_idx, offset):
        r = local_idx // 2
        c = local_idx % 2
        return r * 4 + (offset * 2 + c)
        
    def map_B(local_idx, row_off, col_off):
        r = local_idx // 2
        c = local_idx % 2
        return (row_off * 2 + r) * 4 + (col_off * 2 + c)
        
    def map_C(local_idx, col_off):
        r = local_idx // 2
        c = local_idx % 2
        return (col_off * 2 + c) * 2 + r

    for j, col_def in enumerate(strassen_cols):
        col_idx = j
        for loc_idx, coeff in col_def[0]:
            U[map_A(loc_idx, 0), col_idx] += coeff
        for loc_idx, coeff in col_def[1]:
            V[map_B(loc_idx, 0, 0), col_idx] += coeff
        for loc_idx, coeff in col_def[2]:
            W[map_C(loc_idx, 0), col_idx] += coeff
            
    for j, col_def in enumerate(strassen_cols):
        col_idx = 7 + j
        for loc_idx, coeff in col_def[0]:
            U[map_A(loc_idx, 1), col_idx] += coeff
        for loc_idx, coeff in col_def[1]:
            V[map_B(loc_idx, 1, 0), col_idx] += coeff
        for loc_idx, coeff in col_def[2]:
            W[map_C(loc_idx, 0), col_idx] += coeff
            
    for j, col_def in enumerate(strassen_cols):
        col_idx = 14 + j
        for loc_idx, coeff in col_def[0]:
            U[map_A(loc_idx, 0), col_idx] += coeff
        for loc_idx, coeff in col_def[1]:
            V[map_B(loc_idx, 0, 1), col_idx] += coeff
        for loc_idx, coeff in col_def[2]:
            W[map_C(loc_idx, 1), col_idx] += coeff
            
    for j, col_def in enumerate(strassen_cols):
        col_idx = 21 + j
        for loc_idx, coeff in col_def[0]:
            U[map_A(loc_idx, 1), col_idx] += coeff
        for loc_idx, coeff in col_def[1]:
            V[map_B(loc_idx, 1, 1), col_idx] += coeff
        for loc_idx, coeff in col_def[2]:
            W[map_C(loc_idx, 1), col_idx] += coeff
            
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Attempt to find a lower rank decomposition?
    # Given the complexity and time limit, we stick to the known rank 28.
    # A search for rank 27 is risky and might not converge.
    
    U, V, W = _generate_strassen_decomposition()
    
    # Check time
    if time.time() - start_time > DEADLINE:
        # This shouldn't happen for static generation
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
