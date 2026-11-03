import time
import numpy as np

_DIMS = (2, 4, 6)

def construct():
    n, m, p = _DIMS
    _t0 = time.time()

    # Strassen 7-term formulas as explicit coordinate lists.
    # Each term is a list of (row, col, weight) for the 2x2 sub-block.
    # U side: (i, j) pairs
    _U_terms = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)],
    ]
    # V side: (j, k) pairs
    _V_terms = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (1,1,1)],
    ]
    # W side: (k, i) pairs  -- note W index is k*n+i
    _W_terms = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)],
    ]

    _R = 42
    U_arr = np.zeros((n * m, _R), dtype=np.int64)
    V_arr = np.zeros((m * p, _R), dtype=np.int64)
    W_arr = np.zeros((p * n, _R), dtype=np.int64)

    # Iterate over the 6 block pairs (g for j-block, h for k-block)
    for _g in range(m // 2):
        for _h in range(p // 2):
            _base = (_g * (p // 2) + _h) * 7
            for _s in range(7):
                _c = _base + _s
                # Fill U: rows indexed i*m + j
                for (ri, cj, w) in _U_terms[_s]:
                    U_arr[ri * m + (_g * 2 + cj), _c] = w
                # Fill V: rows indexed j*p + k
                for (rj, ck, w) in _V_terms[_s]:
                    V_arr[(_g * 2 + rj) * p + (_h * 2 + ck), _c] = w
                # Fill W: rows indexed k*n + i
                for (rk, ci, w) in _W_terms[_s]:
                    W_arr[(_h * 2 + rk) * n + ci, _c] = w

    # Quick verification: reconstruct C = A*B and compare with tensor output
    _A = np.random.randint(-3, 4, (n, m)).astype(np.float64)
    _B = np.random.randint(-3, 4, (m, p)).astype(np.float64)
    _C_ref = _A @ _B
    _C_rec = np.zeros((n, p), dtype=np.float64)
    for _c in range(_R):
        _a_vec = _A.reshape(n * m) @ U_arr[:, _c]
        _b_vec = _B.reshape(m * p) @ V_arr[:, _c]
        _C_rec += _a_vec * _b_vec * W_arr[:, _c].reshape(p, n).T
    if not np.allclose(_C_ref, _C_rec, atol=1e-9):
        raise ValueError("Tensor decomposition verification failed")

    return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
