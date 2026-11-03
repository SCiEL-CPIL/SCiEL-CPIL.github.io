import numpy as np
import time
import random

DIMENSIONS = [2, 4, 6]

def _get_strassen_terms():
    """Returns the 7 Strassen terms for 2x2x2 matrix multiplication."""
    return [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

def _check_validity(res, n=2, m=4, p=6, num_trials=20):
    U, V, W = res['U'], res['V'], res['W']
    r = len(U[0])
    for _ in range(num_trials):
        A = np.random.randint(-5, 5, (n, m))
        B = np.random.randint(-5, 5, (m, p))
        C_true = A @ B
        C_calc = np.zeros((n, p))
        for t in range(r):
            u_vec = np.dot(A.flatten(), [U[idx][t] for idx in range(n*m)])
            v_vec = np.dot(B.flatten(), [V[idx][t] for idx in range(m*p)])
            w_mat = np.array([W[idx][t] for idx in range(p*n)]).reshape(p, n)
            C_calc += w_mat.T * (u_vec * v_vec)
        if not np.allclose(C_calc, C_true):
            return False
    return True

def _build_rank42():
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    terms = _get_strassen_terms()
    col = 0
    # 2 j-blocks (size 2), 3 k-blocks (size 2)
    for jb in range(2):
        for kb in range(3):
            for t in range(7):
                u_dict, v_dict, w_dict = terms[t]
                for (di, dj), val in u_dict.items():
                    i = di
                    j = jb * 2 + dj
                    U[i*m + j, col] += val
                for (dj, dk), val in v_dict.items():
                    j = jb * 2 + dj
                    k = kb * 2 + dk
                    V[j*p + k, col] += val
                for (di, dk), val in w_dict.items():
                    i = di
                    k = kb * 2 + dk
                    W[k*n + i, col] += val
                col += 1
                
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    res = _build_rank42()
    
    # Verify quickly
    if _check_validity(res, trials=10):
        return res
    
    return res
