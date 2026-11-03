"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_222():
    """
    Returns U, V, W for 2x2x2 multiplication using Strassen's algorithm (Rank 7).
    """
    n, m, p = 2, 2, 2
    r = 7
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Strassen terms
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _school_221():
    """
    Returns U, V, W for 2x2x1 multiplication using schoolbook (Rank 4).
    """
    n, m, p = 2, 2, 1
    r = 4
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    for c in range(r):
        i = c // m
        j = c % m
        k = 0
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
        
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W satisfy the tensor identity exactly.
    """
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    s = np.dot(W[k*n+i, :], U[i*m+j, :] * V[jp*p+k, :])
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _build_block_decomposition(n, m, p, m_splits, p_splits):
    """
    Builds a decomposition by splitting m and p into blocks.
    m_splits: list of sizes for m
    p_splits: list of tuples (size, type) where type is 'strassen' or 'school'
    """
    U_s, V_s, W_s = _strassen_222()
    U_sc, V_sc, W_sc = _school_221()
    
    total_rank = 0
    # Calculate total rank first to allocate arrays
    for m_size in m_splits:
        for p_size, p_type in p_splits:
            if p_type == 'strassen':
                total_rank += 7
            else:
                total_rank += 4
                
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_offset = 0
    m_offset = 0
    
    for m_size in m_splits:
        p_offset = 0
        for p_size, p_type in p_splits:
            if p_type == 'strassen':
                U_sub, V_sub, W_sub = U_s, V_s, W_s
                r_sub = 7
            else:
                U_sub, V_sub, W_sub = U_sc, V_sc, W_sc
                r_sub = 4
            
            # Place U
            for i in range(n):
                for j_sub in range(m_size):
                    j = m_offset + j_sub
                    U[i * m + j, col_offset:col_offset + r_sub] = U_sub[i * 2 + j_sub, :]
            
            # Place V
            for j_sub in range(m_size):
                j = m_offset + j_sub
                for k_sub in range(p_size):
                    k = p_offset + k_sub
                    V[j * p + k, col_offset:col_offset + r_sub] = V_sub[j_sub * p_size + k_sub, :]
            
            # Place W
            for k_sub in range(p_size):
                k = p_offset + k_sub
                for i in range(n):
                    W[k * n + i, col_offset:col_offset + r_sub] = W_sub[k_sub * 2 + i, :]
            
            col_offset += r_sub
            p_offset += p_size
        m_offset += m_size
        
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Define splits
    # m=4 split into 2, 2
    m_splits = [2, 2]
    # p=7 split into 2, 2, 2, 1
    p_splits = [(2, 'strassen'), (2, 'strassen'), (2, 'strassen'), (1, 'school')]
    
    U, V, W = _build_block_decomposition(n, m, p, m_splits, p_splits)
    
    if time.time() < deadline:
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Fallback
            r = n * m * p
            U_fb = [[0] * r for _ in range(n * m)]
            V_fb = [[0] * r for _ in range(m * p)]
            W_fb = [[0] * r for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][column] = 1
                        V_fb[j * p + k][column] = 1
                        W_fb[k * n + i][column] = 1
                        column += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
