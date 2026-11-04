import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _validate(U, V, W, n=4, m=4, p=5, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    M_decomp = np.einsum('ijr,jkr,kir->ikj', U3, V3, W3)
    
    if not np.allclose(M_decomp, 1.0, atol=tol):
        return False
    return True

def _construct_rank65():
    n, m, p = 4, 4, 5
    rank = 65
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # 1. Generate 49 terms for 4x4x4
    base = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,1,0,0]),
        ([1,0,0,0], [0,0,1,-1], [0,0,1,0]),
        ([0,0,0,1], [1,-1,0,0], [0,0,0,1]),
        ([1,1,0,0], [0,0,0,1], [0,0,1,0]),
        ([-1,0,1,0], [1,1,0,0], [0,1,0,0]),
        ([0,1,0,-1], [0,0,1,1], [0,0,0,1])
    ]
    
    col = 0
    def add_term(u_2x2, v_2x2, w_2x2, a_blocks, a_signs, b_blocks, b_signs, c_blocks, c_signs):
        nonlocal col
        u_4x4 = [0.0] * 16
        v_4x4 = [0.0] * 16
        w_4x4 = [0.0] * 16
        
        for i, (block, sign) in enumerate(zip(a_blocks, a_signs)):
            bi, bj = block
            for r in range(2):
                for c in range(2):
                    idx = (bi*2+r)*4 + (bj*2+c)
                    idx_2x2 = r*2+c
                    u_4x4[idx] += sign * u_2x2[idx_2x2]
                    
        for i, (block, sign) in enumerate(zip(b_blocks, b_signs)):
            bi, bj = block
            for r in range(2):
                for c in range(2):
                    idx = (bi*2+r)*4 + (bj*2+c)
                    idx_2x2 = r*2+c
                    v_4x4[idx] += sign * v_2x2[idx_2x2]
                    
        for i, (block, sign) in enumerate(zip(c_blocks, c_signs)):
            bi, bj = block
            for r in range(2):
                for c in range(2):
                    idx = (bi*2+r)*4 + (bj*2+c)
                    idx_2x2 = r*2+c
                    w_4x4[idx] += sign * w_2x2[idx_2x2]
                    
        U[:, col] = u_4x4
        V[:, col] = v_4x4
        W[:, col] = w_4x4
        col += 1

    # Corrected block indices for W (K, I)
    # M1: C00(0,0), C11(1,1)
    for u, v, w in base:
        add_term(u, v, w, 
                 [(0,0), (1,1)], [1, 1], 
                 [(0,0), (1,1)], [1, 1], 
                 [(0,0), (1,1)], [1, 1])
                 
    # M2: C10(0,1), C11(1,1)
    for u, v, w in base:
        add_term(u, v, w, 
                 [(1,0), (1,1)], [1, 1], 
                 [(0,0)], [1], 
                 [(0,1), (1,1)], [1, -1])
                 
    # M3: C01(1,0), C11(1,1)
    for u, v, w in base:
        add_term(u, v, w, 
                 [(0,0)], [1], 
                 [(0,1), (1,1)], [1, -1], 
                 [(1,0), (1,1)], [1, 1])
                 
    # M4: C00(0,0), C10(0,1)
    for u, v, w in base:
        add_term(u, v, w, 
                 [(1,1)], [1], 
                 [(1,0), (0,0)], [1, -1], 
                 [(0,0), (0,1)], [1, 1])
                 
    # M5: C00(0,0), C01(1,0)
    for u, v, w in base:
        add_term(u, v, w, 
                 [(0,0), (0,1)], [1, 1], 
                 [(1,1)], [1], 
                 [(0,0), (1,0)], [-1, 1])
                 
    # M6: C11(1,1)
    for u, v, w in base:
        add_term(u, v, w, 
                 [(1,0), (0,0)], [1, -1], 
                 [(0,0), (0,1)], [1, 1], 
                 [(1,1)], [1])
                 
    # M7: C00(0,0)
    for u, v, w in base:
        add_term(u, v, w, 
                 [(0,1), (1,1)], [1, -1], 
                 [(1,0), (1,1)], [1, 1], 
                 [(0,0)], [1])
                 
    # 2. Add 16 terms for 4x4x1
    for i in range(4):
        for j in range(4):
            U[i*4 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    try:
        result = _construct_rank65()
        if _validate(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass
        
    # Fallback to 79
    # ... (existing 79 code) ...
    return _construct_strassen_embedded()
