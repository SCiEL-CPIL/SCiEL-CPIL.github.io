import numpy as np
import time

DIMENSIONS = [2, 2, 4]

def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    rank = 14
    U = np.zeros((u_dim, rank))
    V = np.zeros((v_dim, rank))
    W = np.zeros((w_dim, rank))
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i

    # Strassen 7 terms for 2x2x2
    # (U_vec, V_vec, W_vec)
    # Indices: U is (i,j), V is (j,k), W is (k,i)
    
    # 1. (A00+A11)(B00+B11) -> (a+d)(e+j) in current code notation?
    # Let's use the exact terms from the provided working code to ensure validity.
    
    terms = []
    
    # Term 1
    t1_u = [(idx_u(0,0), 1), (idx_u(1,1), 1)]
    t1_v = [(idx_v(0,0), 1), (idx_v(1,1), 1)]
    t1_w = [(idx_w(0,0), 1), (idx_w(1,1), 1)]
    terms.append((t1_u, t1_v, t1_w))
    
    # Term 2
    t2_u = [(idx_u(1,0), 1), (idx_u(1,1), 1)]
    t2_v = [(idx_v(0,0), 1)]
    t2_w = [(idx_w(0,1), 1), (idx_w(1,1), -1)]
    terms.append((t2_u, t2_v, t2_w))
    
    # Term 3
    t3_u = [(idx_u(0,0), 1)]
    t3_v = [(idx_v(0,1), 1), (idx_v(1,1), -1)]
    t3_w = [(idx_w(1,0), 1), (idx_w(1,1), 1)]
    terms.append((t3_u, t3_v, t3_w))
    
    # Term 4
    t4_u = [(idx_u(1,1), 1)]
    t4_v = [(idx_v(1,0), 1), (idx_v(0,0), -1)]
    t4_w = [(idx_w(0,0), 1), (idx_w(0,1), 1)]
    terms.append((t4_u, t4_v, t4_w))
    
    # Term 5
    t5_u = [(idx_u(0,0), 1), (idx_u(0,1), 1)]
    t5_v = [(idx_v(1,1), 1)]
    t5_w = [(idx_w(1,0), 1), (idx_w(0,0), -1)]
    terms.append((t5_u, t5_v, t5_w))
    
    # Term 6
    t6_u = [(idx_u(1,0), 1), (idx_u(0,0), -1)]
    t6_v = [(idx_v(0,0), 1), (idx_v(0,1), 1)]
    t6_w = [(idx_w(1,1), 1)]
    terms.append((t6_u, t6_v, t6_w))
    
    # Term 7
    t7_u = [(idx_u(0,1), 1), (idx_u(1,1), -1)]
    t7_v = [(idx_v(1,0), 1), (idx_v(1,1), 1)]
    t7_w = [(idx_w(0,0), 1)]
    terms.append((t7_u, t7_v, t7_w))
    
    # Fill matrices
    for r in range(7):
        u_vec, v_vec, w_vec = terms[r]
        
        # Block 1: columns 0, 1 of B (indices 0,1,4,5 in V)
        col = r
        for idx, val in u_vec:
            U[idx, col] += val
        for idx, val in v_vec:
            V[idx, col] += val
        for idx, val in w_vec:
            W[idx, col] += val
            
        # Block 2: columns 2, 3 of B (indices 2,3,6,7 in V)
        col = r + 7
        for idx, val in u_vec:
            U[idx, col] += val
            
        # Shift V indices: k -> k+2
        for idx, val in v_vec:
            V[idx + 2, col] += val
            
        # Shift W indices: k -> k+2 (output cols 2,3)
        for idx, val in w_vec:
            W[idx + 4, col] += val
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
