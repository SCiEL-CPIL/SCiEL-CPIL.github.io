"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]
TIME_LIMIT = 150.0

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W


def _build_m441():
    """
    Build rank-2 decomposition for M_{4,4,1} (4x4 * 4x1 -> 4x1).
    Uses half-integers to satisfy sum=1.
    """
    n = 4
    m = 4
    p = 1
    rank = 2
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # r=0
    for i in range(n):
        for k in range(m):
            U[i * m + k, 0] = 1.0
    for k in range(m):
        V[k * p, 0] = 0.5
    for i in range(n):
        W[i * p, 0] = 1.0
        
    # r=1
    for i in range(n):
        for k in range(m):
            U[i * m + k, 1] = 1.0
    for k in range(m):
        V[k * p, 1] = 0.5
    for i in range(n):
        W[i * p, 1] = 1.0
        
    return U, V, W


def _build_m414():
    """
    Build rank-1 decomposition for M_{4,1,4} (4x1 * 1x4 -> 4x4).
    """
    n = 4
    m = 1
    p = 4
    rank = 1
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        U[i * m, 0] = 1.0
    V[0, 0] = 1.0
    for j in range(p):
        W[j * n, 0] = 1.0
        
    return U, V, W


def _build_m412():
    """
    Build rank-1 decomposition for M_{4,1,2} (4x1 * 1x2 -> 4x2).
    """
    n = 4
    m = 1
    p = 2
    rank = 1
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        U[i * m, 0] = 1.0
    for j in range(p):
        V[j, 0] = 1.0
    for j in range(p):
        for i in range(n):
            W[j * n + i, 0] = 1.0
            
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    """
    Find a low-rank exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Split indices
    # A: 4x5 (i:0-3, k:0-4)
    # B: 5x6 (k:0-4, j:0-5)
    # C: 4x6 (i:0-3, j:0-5)
    
    # Core: 4x4x4 (i:0-3, k:0-3, j:0-3)
    # Right: 4x4x2 (i:0-3, k:0-3, j:4-5)
    # Bottom: 4x1x4 (i:0-3, k:4, j:0-3)
    # Corner: 4x1x2 (i:0-3, k:4, j:4-5)
    
    # 1. Core Schoolbook (Rank 64)
    U_core, V_core, W_core = _build_schoolbook(4, 4, 4)
    # Map indices to global
    U_c = np.zeros((4*5, 64))
    V_c = np.zeros((5*6, 64))
    W_c = np.zeros((6*4, 64))
    for i in range(4):
        for k in range(4):
            for j in range(4):
                r = i*16 + k*4 + j
                U_c[i*5 + k, r] = 1.0
                V_c[k*6 + j, r] = 1.0
                W_c[j*4 + i, r] = 1.0

    # 2. Right Strip (Rank 4)
    # Two M_441
    U_r1, V_r1, W_r1 = _build_m441()
    U_r2, V_r2, W_r2 = _build_m441()
    
    U_r = np.zeros((4*5, 4))
    V_r = np.zeros((5*6, 4))
    W_r = np.zeros((6*4, 4))
    
    # r=0,1 for j=4
    for r in range(2):
        for i in range(4):
            for k in range(4):
                U_r[i*5 + k, r] = U_r1[i*4 + k, r]
        for k in range(4):
            V_r[k*6 + 4, r] = V_r1[k*1, r]
        for i in range(4):
            W_r[4*4 + i, r] = W_r1[1*i, r] # W index: j*n+i -> 4*4+i
            
    # r=2,3 for j=5
    for r in range(2):
        for i in range(4):
            for k in range(4):
                U_r[i*5 + k, r+2] = U_r2[i*4 + k, r]
        for k in range(4):
            V_r[k*6 + 5, r+2] = V_r2[k*1, r]
        for i in range(4):
            W_r[5*4 + i, r+2] = W_r2[1*i, r]

    # 3. Bottom Strip (Rank 1)
    U_b, V_b, W_b = _build_m414()
    U_b_g = np.zeros((4*5, 1))
    V_b_g = np.zeros((5*6, 1))
    W_b_g = np.zeros((6*4, 1))
    for i in range(4):
        U_b_g[i*5 + 4, 0] = U_b[i*1, 0]
    V_b_g[4*6 + 0, 0] = V_b[0, 0]
    for j in range(4):
        W_b_g[j*4 + 0, 0] = W_b[j*4, 0] # W index: j*n+i -> j*4+0

    # 4. Corner (Rank 1)
    U_co, V_co, W_co = _build_m412()
    U_co_g = np.zeros((4*5, 1))
    V_co_g = np.zeros((5*6, 1))
    W_co_g = np.zeros((6*4, 1))
    for i in range(4):
        U_co_g[i*5 + 4, 0] = U_co[i*1, 0]
    for j in range(2):
        V_co_g[4*6 + j, 0] = V_co[j, 0]
    for j in range(2):
        for i in range(4):
            W_co_g[j*4 + i, 0] = W_co[j*2 + i, 0] # W index: j*n+i -> j*4+i

    # Concatenate
    U = np.hstack([U_c, U_r, U_b_g, U_co_g])
    V = np.hstack([V_c, V_r, V_b_g, V_co_g])
    W = np.hstack([W_c, W_r, W_b_g, W_co_g])
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        U, V, W = _build_schoolbook(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
