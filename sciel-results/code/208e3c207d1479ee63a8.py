import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    strassen_blocks = [
        {'u': [(0, 1), (3, 1)], 'v': [(0, 1), (3, 1)], 'w': [(0, 1), (3, 1)]},
        {'u': [(2, 1), (3, 1)], 'v': [(0, 1)], 'w': [(2, 1), (3, -1)]},
        {'u': [(0, 1)], 'v': [(1, 1), (3, -1)], 'w': [(1, 1), (3, 1)]},
        {'u': [(3, 1)], 'v': [(2, 1), (0, -1)], 'w': [(0, 1), (2, 1)]},
        {'u': [(0, 1), (1, 1)], 'v': [(3, 1)], 'w': [(0, -1), (1, 1)]},
        {'u': [(2, 1), (0, -1)], 'v': [(0, 1), (1, 1)], 'w': [(3, 1)]},
        {'u': [(1, 1), (3, -1)], 'v': [(2, 1), (3, 1)], 'w': [(0, 1)]}
    ]
    
    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # 8 blocks: 2 column-pairs of A, 4 column-pairs of B
    for u in range(2):
        for v in range(4):
            a_ids = [0 * m + 2 * u, 0 * m + 2 * u + 1, 1 * m + 2 * u, 1 * m + 2 * u + 1]
            b_ids = [(2 * u) * p + 2 * v, (2 * u) * p + 2 * v + 1, (2 * u + 1) * p + 2 * v, (2 * u + 1) * p + 2 * v + 1]
            c_ids = [(2 * v) * n + 0, (2 * v + 1) * n + 0, (2 * v) * n + 1, (2 * v + 1) * n + 1]
            
            base_col = 7 * (4 * u + v)
            
            for s_idx, pattern in enumerate(strassen_blocks):
                col = base_col + s_idx
                
                for idx, coeff in pattern['u']:
                    U[a_ids[idx], col] += coeff
                for idx, coeff in pattern['v']:
                    V[b_ids[idx], col] += coeff
                for idx, coeff in pattern['w']:
                    W[c_ids[idx], col] += coeff

    # Verify base solution
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
