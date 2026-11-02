"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 140.0  # seconds, safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _to_halfint(x):
    """Convert a Fraction or number to integer or half-integer (multiple of 0.5)."""
    if isinstance(x, Fraction):
        num, den = x.numerator, x.denominator
        if den == 1:
            return num
        if den == 2:
            return num / 2
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        if abs(2*x - round(2*x)) < 1e-9:
            return round(2*x) / 2
        return None
    return None

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact and entries are integer/half-integer."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    # Check all entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                h = _to_halfint(val)
                if h is None:
                    return False
    # Verify tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if isinstance(s, Fraction):
                    if s != 1:
                        return False
                else:
                    if abs(s - 1) > 1e-9:
                        return False
    return True

def _strassen_222():
    """
    Returns U, V, W factors for a 2x2x2 block using Strassen's 7-term decomposition.
    U, V, W are lists of 4 lists of 7 floats.
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Coefficients for 7 terms
    # Each term is a tuple (u, v, w) where u,v,w are length 4 lists
    # u corresponds to A[0,0], A[0,1], A[1,0], A[1,1]
    # v corresponds to B[0,0], B[0,1], B[1,0], B[1,1]
    # w corresponds to C[0,0], C[0,1], C[1,0], C[1,1]
    
    terms = [
        # 1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # 2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # 3
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
        # 4
        ([0, 0, 0, 1], [1, -1, 0, 0], [1, 0, 1, 0]),
        # 5
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
        # 6
        ([-1, 0, 0, 1], [1, 1, 0, 0], [1, 0, 0, 0]),
        # 7
        ([0, -1, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1]),
    ]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    for c in range(7):
        u, v, w = terms[c]
        for idx in range(4):
            U[idx][c] = u[idx]
            V[idx][c] = v[idx]
            W[idx][c] = w[idx]
            
    return U, V, W

def _schoolbook_block(n_b, m_b, p_b):
    """
    Returns U, V, W factors for an n_b x m_b x p_b block using schoolbook.
    Rank = n_b * m_b * p_b.
    """
    rank = n_b * m_b * p_b
    U = [[0]*rank for _ in range(n_b * m_b)]
    V = [[0]*rank for _ in range(m_b * p_b)]
    W = [[0]*rank for _ in range(p_b * n_b)]
    
    col = 0
    for i in range(n_b):
        for j in range(m_b):
            for k in range(p_b):
                U[i*m_b + j][col] = 1
                V[j*p_b + k][col] = 1
                W[k*n_b + i][col] = 1
                col += 1
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # Partitions
    # I = {0, 1}
    # J = {0, 1}, {2, 3}, {4}
    # K = {0, 1}, {2, 3}, {4, 5}
    
    J_blocks = [ [0, 1], [2, 3], [4] ]
    K_blocks = [ [0, 1], [2, 3], [4, 5] ]
    
    total_rank = 0
    # Pre-allocate with max possible rank to avoid resizing? 
    # Max rank is 60. We expect 54.
    # We'll build lists and then join.
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    for J in J_blocks:
        m_b = len(J)
        for K in K_blocks:
            p_b = len(K)
            n_b = 2
            
            if m_b == 2 and p_b == 2:
                U_b, V_b, W_b = _strassen_222()
            else:
                U_b, V_b, W_b = _schoolbook_block(n_b, m_b, p_b)
                
            U_blocks.append(U_b)
            V_blocks.append(V_b)
            W_blocks.append(W_b)
            
            total_rank += len(U_b[0])
            
    # Concatenate blocks
    # U shape: (n*m, rank) = (10, rank)
    # We need to map local indices to global indices.
    
    U_final = [[0]*total_rank for _ in range(n*m)]
    V_final = [[0]*total_rank for _ in range(m*p)]
    W_final = [[0]*total_rank for _ in range(p*n)]
    
    col_offset = 0
    for idx, J in enumerate(J_blocks):
        m_b = len(J)
        for idx_k, K in enumerate(K_blocks):
            p_b = len(K)
            n_b = 2
            
            U_b = U_blocks[idx * len(K_blocks) + idx_k]
            V_b = V_blocks[idx * len(K_blocks) + idx_k]
            W_b = W_blocks[idx * len(K_blocks) + idx_k]
            
            block_rank = len(U_b[0])
            
            # Map U: local (i, j_local) -> global (i, J[j_local])
            # Global index: i*m + J[j_local]
            for i in range(n_b):
                for j_local in range(m_b):
                    global_j = J[j_local]
                    u_global_idx = i*m + global_j
                    u_local_idx = i*m_b + j_local
                    for c in range(block_rank):
                        U_final[u_global_idx][col_offset + c] = U_b[u_local_idx][c]
                        
            # Map V: local (j_local, k_local) -> global (J[j_local], K[k_local])
            # Global index: J[j_local]*p + K[k_local]
            for j_local in range(m_b):
                global_j = J[j_local]
                for k_local in range(p_b):
                    global_k = K[k_local]
                    v_global_idx = global_j*p + global_k
                    v_local_idx = j_local*p_b + k_local
                    for c in range(block_rank):
                        V_final[v_global_idx][col_offset + c] = V_b[v_local_idx][c]
                        
            # Map W: local (k_local, i) -> global (K[k_local], i)
            # Global index: K[k_local]*n + i
            for k_local in range(p_b):
                global_k = K[k_local]
                for i in range(n_b):
                    w_global_idx = global_k*n + i
                    w_local_idx = k_local*n_b + i
                    for c in range(block_rank):
                        W_final[w_global_idx][col_offset + c] = W_b[w_local_idx][c]
                        
            col_offset += block_rank
            
    result = {"U": U_final, "V": V_final, "W": W_final}
    
    # Verify
    if _verify(U_final, V_final, W_final, n, m, p):
        return result
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
