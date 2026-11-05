import numpy as np
import time
from fractions import Fraction
import sys

START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin for 120s limit

def _build_trivial(n, m, p):
    """Build the trivial schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1.0
                V[j*p + k, idx] = 1.0
                W[k*n + i, idx] = 1.0
                idx += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, num_trials=20):
    """Verify that the decomposition is correct by random testing."""
    rng = np.random.default_rng(42)
    for _ in range(num_trials):
        A = rng.random((n, m))
        B = rng.random((m, p))
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            u_vec = U[:, r]
            v_vec = V[:, r]
            w_val = W[:, r]
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            prod = u_mat @ v_mat
            w_mat = w_val.reshape(p, n)
            C_computed += prod * w_mat.T
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def _append(U, V, W, u, v, w):
    U = np.column_stack((U, u)) if U.shape[1] > 0 else u.reshape(-1, 1)
    V = np.column_stack((V, v)) if V.shape[1] > 0 else v.reshape(-1, 1)
    W = np.column_stack((W, w)) if W.shape[1] > 0 else w.reshape(-1, 1)
    return U, V, W

def _build_strassen_2x2x2(U, V, W, i_off, j_off, k_off, n, m, p):
    u1 = np.array([1, 0, 0, 1])
    v1 = np.array([1, 1, 1, 1])
    w1 = np.zeros(4); w1[0] = 1; w1[3] = 1
    
    u2 = np.array([0, 0, 1, 1])
    v2 = np.array([1, 0, 0, 0])
    w2 = np.zeros(4); w2[2] = 1; w2[3] = -1
    
    u3 = np.array([1, 0, 0, 0])
    v3 = np.array([0, 1, 0, -1])
    w3 = np.zeros(4); w3[1] = 1; w3[3] = 1
    
    u4 = np.array([0, 0, 0, 1])
    v4 = np.array([0, 0, 1, -1])
    w4 = np.zeros(4); w4[0] = 1; w4[2] = 1
    
    u5 = np.array([1, 1, 0, 0])
    v5 = np.array([0, 0, 0, 1])
    w5 = np.zeros(4); w5[0] = -1; w5[1] = 1
    
    u6 = np.array([-1, 0, 1, 0])
    v6 = np.array([1, 1, 0, 0])
    w6 = np.zeros(4); w6[3] = 1
    
    u7 = np.array([0, 1, 0, -1])
    v7 = np.array([0, 0, 1, 1])
    w7 = np.zeros(4); w7[0] = 1
    
    terms = [(u1, v1, w1), (u2, v2, w2), (u3, v3, w3), (u4, v4, w4), (u5, v5, w5), (u6, v6, w6), (u7, v7, w7)]
    
    for u, v, w in terms:
        u_global = np.zeros(n*m)
        v_global = np.zeros(m*p)
        w_global = np.zeros(p*n)
        
        for li in range(2):
            for lj in range(2):
                val = u[li*2 + lj]
                if val != 0:
                    gi = i_off + li
                    gj = j_off + lj
                    u_global[gi*m + gj] = val
                    
        for lj in range(2):
            for lk in range(2):
                val = v[lj*2 + lk]
                if val != 0:
                    gj = j_off + lj
                    gk = k_off + lk
                    v_global[gj*p + gk] = val
                    
        for lk in range(2):
            for li in range(2):
                val = w[lk*2 + li]
                if val != 0:
                    gk = k_off + lk
                    gi = i_off + li
                    w_global[gk*n + gi] = val
                    
        U, V, W = _append(U, V, W, u_global, v_global, w_global)
        
    return U, V, W

def _build_schoolbook_block(U, V, W, i_off, j_off, k_off, n, m, p, ni, nj, nk):
    for li in range(ni):
        for lj in range(nj):
            for lk in range(nk):
                u_global = np.zeros(n*m)
                v_global = np.zeros(m*p)
                w_global = np.zeros(p*n)
                
                gi = i_off + li
                gj = j_off + lj
                gk = k_off + lk
                
                u_global[gi*m + gj] = 1.0
                v_global[gj*p + gk] = 1.0
                w_global[gk*n + gi] = 1.0
                
                U, V, W = _append(U, V, W, u_global, v_global, w_global)
                
    return U, V, W

def construct():
    n, m, p = 3, 5, 7
    
    U = np.zeros((n*m, 0))
    V = np.zeros((m*p, 0))
    W = np.zeros((p*n, 0))
    
    # 1. i=2, j=0..4, k=0..6 (1x5 by 5x7)
    U, V, W = _build_schoolbook_block(U, V, W, 2, 0, 0, n, m, p, 1, 5, 7)
    
    # 2. i=0..1
    # 2a. j=4, k=0..6 (2x1 by 1x7)
    U, V, W = _build_schoolbook_block(U, V, W, 0, 4, 0, n, m, p, 2, 1, 7)
    
    # 2b. j=0..1 (2x2 by 2x7)
    # 2b-i. k=0..1 (2x2)
    U, V, W = _build_strassen_2x2x2(U, V, W, 0, 0, 0, n, m, p)
    # 2b-ii. k=2..3 (2x2)
    U, V, W = _build_strassen_2x2x2(U, V, W, 0, 0, 2, n, m, p)
    # 2b-iii. k=4..5 (2x2)
    U, V, W = _build_strassen_2x2x2(U, V, W, 0, 0, 4, n, m, p)
    # 2b-iv. k=6..6 (2x1)
    U, V, W = _build_schoolbook_block(U, V, W, 0, 0, 6, n, m, p, 2, 2, 1)
    
    # 2c. j=2..3 (2x2 by 2x7)
    # 2c-i. k=0..1 (2x2)
    U, V, W = _build_strassen_2x2x2(U, V, W, 0, 2, 0, n, m, p)
    # 2c-ii. k=2..3 (2x2)
    U, V, W = _build_strassen_2x2x2(U, V, W, 0, 2, 2, n, m, p)
    # 2c-iii. k=4..5 (2x2)
    U, V, W = _build_strassen_2x2x2(U, V, W, 0, 2, 4, n, m, p)
    # 2c-iv. k=6..6 (2x1)
    U, V, W = _build_schoolbook_block(U, V, W, 0, 2, 6, n, m, p, 2, 2, 1)
    
    if not _verify_decomposition(U, V, W, n, m, p, num_trials=5):
        U, V, W = _build_trivial(n, m, p)
        
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
