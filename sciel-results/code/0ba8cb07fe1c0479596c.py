import numpy as np
import time

DIMENSIONS = [3, 5, 6]
DEADLINE = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0

def _snap_to_half(x):
    return round(2 * x) / 2.0

def _verify_fast(U, V, W, n, m, p):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    if Ua.shape != (n*m, rank) or Va.shape != (m*p, rank) or Wa.shape != (p*n, rank):
        return False
    
    for arr in [Ua, Va, Wa]:
        if not np.allclose(2*arr, np.round(2*arr), atol=1e-6):
            return False
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-6):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_222():
    # Returns U, V, W for 2x2x2 Strassen
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # r=2: M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # r=3: M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # r=4: M5
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # r=5: M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # r=6: M7
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[1, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _construct_block_decomposition(n, m, p):
    # Returns U, V, W for the whole problem using block decomposition
    U = np.zeros((n*m, 84))
    V = np.zeros((m*p, 84))
    W = np.zeros((p*n, 84))
    
    rank_offset = 0
    
    # Define blocks: (i_indices, j_indices, k_indices, use_strassen)
    # We need to map local indices to global indices
    
    # Block 1-3: I1={0,1}, J1={0,1}, K1={0,1}, K2={2,3}, K3={4,5}
    # Strassen
    
    # Helper to add a block
    def add_block(i_idx, j_idx, k_idx, is_strassen):
        nonlocal rank_offset
        n_b = len(i_idx)
        m_b = len(j_idx)
        p_b = len(k_idx)
        
        if is_strassen:
            if n_b != 2 or m_b != 2 or p_b != 2:
                raise ValueError("Strassen requires 2x2x2")
            Ub, Vb, Wb = _strassen_222()
            r_b = 7
        else:
            Ub, Vb, Wb = _schoolbook_block(n_b, m_b, p_b)
            r_b = n_b * m_b * p_b
            
        # Map rows
        for r in range(r_b):
            for i_local, i_global in enumerate(i_idx):
                for j_local, j_global in enumerate(j_idx):
                    U[i_global*m + j_global, rank_offset + r] += Ub[i_local*m_b + j_local, r]
                    
            for j_local, j_global in enumerate(j_idx):
                for k_local, k_global in enumerate(k_idx):
                    V[j_global*p + k_global, rank_offset + r] += Vb[j_local*p_b + k_local, r]
                    
            for k_local, k_global in enumerate(k_idx):
                for i_local, i_global in enumerate(i_idx):
                    W[k_global*n + i_global, rank_offset + r] += Wb[k_local*n_b + i_local, r]
                    
        rank_offset += r_b

    # Partition indices
    I1 = [0, 1]
    I2 = [2]
    J1 = [0, 1]
    J2a = [2, 3]
    J2b = [4]
    K1 = [0, 1]
    K2 = [2, 3]
    K3 = [4, 5]
    
    # Blocks for J1
    add_block(I1, J1, K1, True)
    add_block(I1, J1, K2, True)
    add_block(I1, J1, K3, True)
    add_block(I2, J1, K1, False)
    add_block(I2, J1, K2, False)
    add_block(I2, J1, K3, False)
    
    # Blocks for J2 (split into J2a, J2b)
    add_block(I1, J2a, K1, True)
    add_block(I1, J2b, K1, False)
    add_block(I1, J2a, K2, True)
    add_block(I1, J2b, K2, False)
    add_block(I1, J2a, K3, True)
    add_block(I1, J2b, K3, False)
    
    add_block(I2, J2a, K1, False)
    add_block(I2, J2b, K1, False)
    add_block(I2, J2a, K2, False)
    add_block(I2, J2b, K2, False)
    add_block(I2, J2a, K3, False)
    add_block(I2, J2b, K3, False)
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    
    # Try the block Strassen decomposition first
    U, V, W = _construct_block_decomposition(n, m, p)
    
    result = {"U": U, "V": V, "W": W}
    
    if _verify_fast(U, V, W, n, m, p):
        return result
    
    # Fallback to schoolbook if something went wrong
    return _schoolbook(n, m, p)
