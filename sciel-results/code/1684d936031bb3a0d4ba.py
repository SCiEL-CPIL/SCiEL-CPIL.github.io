import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def check(U, V, W, tol=1e-9):
        if time.time() > deadline:
            return False
        rng = np.random.default_rng(42)
        for _ in range(50):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C = A @ B
            
            L = A.flatten() @ U
            R = B.flatten() @ V
            C_calc = (L * R) @ W.T
            C_calc = C_calc.reshape(p, n).T
            
            if not np.allclose(C_calc, C, atol=tol):
                return False
        return True

    def to_half_int_list(arr):
        res = []
        for row in arr:
            r = []
            for val in row:
                val = round(val * 2) / 2.0
                r.append(val)
            res.append(r)
        return res

    # Schoolbook
    rank_schoolbook = n * m * p
    U = np.zeros((nm, rank_schoolbook))
    V = np.zeros((mp, rank_schoolbook))
    W = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()

    # Strassen 2x2x2 Rank 7
    # Verified matrices
    U_s = np.array([
        [1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 1, 0, 0]
    ])
    V_s = np.array([
        [1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 0]
    ])
    W_s = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0]
    ])
    
    # Verify Strassen block
    def check_strassen_block():
        rng = np.random.default_rng(123)
        for _ in range(100):
            A = rng.standard_normal((2, 2))
            B = rng.standard_normal((2, 2))
            C = A @ B
            L = A.flatten() @ U_s
            R = B.flatten() @ V_s
            C_calc = (L * R) @ W_s.T
            C_calc = C_calc.reshape(2, 2).T
            if not np.allclose(C_calc, C, atol=1e-9):
                return False
        return True

    if check_strassen_block():
        # Construct full 2x2x16 decomposition
        num_blocks = p // 2
        rank = num_blocks * 7
        U_full = np.zeros((nm, rank))
        V_full = np.zeros((mp, rank))
        W_full = np.zeros((pn, rank))
        
        for l in range(num_blocks):
            col_start = l * 7
            col_end = col_start + 7
            
            # U is same for all blocks (A is shared)
            U_full[:, col_start:col_end] = U_s
            
            # V: rows j*16 + k
            for j in range(m):
                # k = 2l
                V_full[j*16 + 2*l, col_start:col_end] = V_s[j, :]
                # k = 2l+1
                V_full[j*16 + 2*l + 1, col_start:col_end] = V_s[j, :]
            
            # W: rows k*2 + i
            for i in range(n):
                # k = 2l
                W_full[2*l*2 + i, col_start:col_end] = W_s[i, :]
                # k = 2l+1
                W_full[(2*l+1)*2 + i, col_start:col_end] = W_s[i, :]
        
        if check(U_full, V_full, W_full):
            best_rank = rank
            best_U = to_half_int_list(U_full)
            best_V = to_half_int_list(V_full)
            best_W = to_half_int_list(W_full)

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
