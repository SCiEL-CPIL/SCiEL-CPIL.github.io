import numpy as np
import time

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def _ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n
    rank = 44

    def _schoolbook():
        # Fallback to full schoolbook rank 48 if optimization fails or is invalid
        R = nm * mp * pn
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        c = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a, c] = 1
                    V[a * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if not _ok():
        return _schoolbook()

    # Strassen 2x2x2 core matrices
    # Indices for U: i, a (0..1) -> row i*2+a
    # Indices for V: a, k (0..1) -> row a*2+k
    # Indices for W: k, i (0..1) -> row k*2+i
    
    S_U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)
    
    S_V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    
    S_W = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=float)

    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    col = 0
    
    # Part 1: 4 Strassen blocks for A[0:2, 0:2] * B[0:2, :]
    # A rows: i=0,1. Cols a=0,1.
    # B rows: a=0,1. Cols k=0..7.
    # We split B's k dimension into 4 blocks of 2.
    
    for block_idx in range(4):
        if not _ok():
            return _schoolbook()
        
        k0 = block_idx * 2
        
        # Assign U: Rows i*3+a for i in 0,1 and a in 0,1
        # U indices: 0,1,3,4 (i=0,a=0; i=0,a=1; i=1,a=0; i=1,a=1)
        u_rows = [0, 1, 3, 4]
        # S_U is 7x4. We assign S_U[:, j] to U[u_rows[j], col:col+7]
        # Looping over the 7 rank terms is fine, but let's use slicing if possible.
        # Since U is flat, we can just assign the column block.
        
        # For each rank term r in 0..6:
        # U[u_rows[j], col+r] = S_U[r, j]
        
        # Vectorized assignment for U part
        # Create a 7x4 matrix, we want to scatter it.
        # Easier: loop over the 7 steps, it's small.
        for r in range(7):
            c = col + r
            # U
            for j, u_idx in enumerate(u_rows):
                val = S_U[r, j]
                if val != 0:
                    U[u_idx, c] = val
            
            # V: Rows a*8+k for a in 0,1 and k in k0, k0+1
            # v_rows relative to block: a*2+k_rel
            # Global V index: a*p + (k0 + k_rel)
            # S_V is 7x4. Col j corresponds to v_idx_j.
            v_rows_global = []
            for a in range(2):
                for k_rel in range(2):
                    # S_V col index: a*2 + k_rel
                    idx = a * 2 + k_rel
                    g_idx = a * p + (k0 + k_rel)
                    v_rows_global.append((idx, g_idx))
            
            for s_v_idx, g_v_idx in v_rows_global:
                val = S_V[r, s_v_idx]
                if val != 0:
                    V[g_v_idx, c] = val
            
            # W: Rows k*n+i for k in k0, k0+1 and i in 0,1
            # S_W col index: k_rel*2 + i
            w_rows_global = []
            for k_rel in range(2):
                for i in range(2):
                    idx = k_rel * 2 + i
                    g_idx = (k0 + k_rel) * n + i
                    w_rows_global.append((idx, g_idx))
            
            for s_w_idx, g_w_idx in w_rows_global:
                val = S_W[r, s_w_idx]
                if val != 0:
                    W[g_w_idx, c] = val
        
        col += 7

    if not _ok():
        return _schoolbook()

    # Part 2: Schoolbook for A[0:2, 2] * B[2, :8]
    # i in 0,1; a=2; k in 0..7
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
            if not _ok():
                return _schoolbook()

    # Verify
    # Check if sum_r U[ia, r] V[ak, r] W[ki, r] == 1
    
    # Optimization: 
    # For each i, a, k:
    # val = np.dot(U[i*m+a, :], V[a*p+k, :] * W[k*n+i, :])
    
    for i in range(n):
        for a in range(m):
            for k in range(p):
                u_vec = U[i * m + a, :]
                v_vec = V[a * p + k, :]
                w_vec = W[k * n + i, :]
                
                # The term is sum(u*r * v*r * w*r)
                # This is dot(u, v * w)
                prod_vw = v_vec * w_vec
                s = np.dot(u_vec, prod_vw)
                
                if abs(s - 1.0) > 1e-9:
                    # Verification failed, fall back
                    return _schoolbook()
    
    if not _ok():
        return _schoolbook()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
