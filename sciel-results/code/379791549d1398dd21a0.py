import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]

def _check_decomp(U, V, W, n, m, p):
    """
    Check if U, V, W exactly decompose the tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    """
    r = U.shape[1]
    # Reshape
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    
    # Check T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for t in range(r):
                    val += U_r[i, j, t] * V_r[j, k, t] * W_r[k, i, t]
                if val != 1:
                    return False
    return True

def _search_decomp_2_3_2(r_target, time_limit=5.0):
    """
    Search for a rank r_target decomposition of M_2,3,2.
    Returns U, V, W or None.
    """
    n, m, p = 2, 3, 2
    start = time.time()
    
    # Random restarts
    for _ in range(1000):
        if time.time() - start > time_limit:
            break
        U = np.random.randint(-2, 3, size=(n*m, r_target))
        V = np.random.randint(-2, 3, size=(m*p, r_target))
        W = np.random.randint(-2, 3, size=(p*n, r_target))
        
        # Simple local improvement?
        # Just check
        if _check_decomp(U, V, W, n, m, p):
            return U, V, W
            
    return None

def _construct_rank_50():
    """
    Try to construct rank 50 decomposition by combining 5 rank 10 blocks.
    """
    res = _search_decomp_2_3_2(10, time_limit=10.0)
    if res is not None:
        U_block, V_block, W_block = res
        n, m, p = 2, 3, 10
        r_total = 50
        U = np.zeros((n*m, r_total))
        V = np.zeros((m*p, r_total))
        W = np.zeros((p*n, r_total))
        
        # 5 blocks for k=0..4 (pairs of columns)
        # Block b handles columns 2b, 2b+1
        for b in range(5):
            offset = b * 10
            U[:, offset:offset+10] = U_block
            # V and W need to be permuted/adjusted for the specific k indices
            # V_block is for m*p=6 (j=0..2, k=0..1)
            # We need V for j=0..2, k=2b..2b+1
            for j in range(m):
                for k_local in range(2):
                    k_global = 2*b + k_local
                    V[j*p + k_global, offset:offset+10] = V_block[j*2 + k_local, :]
            # W_block is for p*n=4 (k=0..1, i=0..1)
            # We need W for k=2b..2b+1, i=0..1
            for k_local in range(2):
                for i in range(n):
                    k_global = 2*b + k_local
                    W[k_global*n + i, offset:offset+10] = W_block[k_local*n + i, :]
        
        if _check_decomp(U, V, W, n, m, p):
            return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 135.0
    
    # Try rank 50
    if time.time() < DEADLINE - 10:
        res = _construct_rank_50()
        if res is not None:
            U, V, W = res
            U = np.rint(U).astype(int)
            V = np.rint(V).astype(int)
            W = np.rint(W).astype(int)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to naive rank 60
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    for j in range(m):
        for i in range(n):
            for k in range(p):
                c = i * m + j
                d = j * p + k
                e = k * n + i
                idx = j * (n * p) + i * p + k
                U[c, idx] = 1.0
                V[d, idx] = 1.0
                W[e, idx] = 1.0
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
