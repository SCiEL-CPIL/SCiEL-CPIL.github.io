import time
import numpy as np

_DIMS = (2, 4, 4)
_T0 = time.time()
_LIMIT = 115  # seconds, safety margin before 120s kill

def _elapsed():
    return time.time() - _T0

def _time_ok(margin=3):
    return _elapsed() < _LIMIT - margin

def _strassen_2x2_patterns():
    return [
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 3: 1}},
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: -1}},
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {0: 1, 2: 1}},
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {0: -1, 1: 1}},
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {3: 1}},
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {0: 1}},
    ]

def _construct_strassen_28():
    n, m, p = _DIMS
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    patterns = _strassen_2x2_patterns()
    blocks = [
        (0, 0, 0, 0),
        (2, 2, 0, 0),
        (0, 0, 2, 2),
        (2, 2, 2, 2),
    ]
    
    col = 0
    for (a_cj, b_rj, b_ck, c_ck) in blocks:
        for pat in patterns:
            for li, coeff in pat['a'].items():
                lr = li // 2
                lc = li % 2
                gi = lr
                gj = a_cj + lc
                U[gi * m + gj, col] = coeff
            
            for li, coeff in pat['b'].items():
                lr = li // 2
                lc = li % 2
                gj = b_rj + lr
                gk = b_ck + lc
                V[gj * p + gk, col] = coeff
            
            for li, coeff in pat['c'].items():
                lr = li // 2
                lc = li % 2
                gi = lr
                gk = c_ck + lc
                W[gk * n + gi, col] = coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}

def _construct_schoolbook():
    n, m, p = _DIMS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    if not _time_ok(margin=2):
        return _construct_schoolbook()
    return _construct_strassen_28()
