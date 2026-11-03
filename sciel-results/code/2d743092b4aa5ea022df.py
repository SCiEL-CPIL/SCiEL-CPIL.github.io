import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _check_exact(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all (i,j,k)."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            v_base = j * p
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += u_row[c] * V[v_base + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2x2 matrix multiplication with rank 7.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, -1, 0]
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 0]
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, -1]
    ], dtype=float)
    
    if not _check_exact(U, V, W, 2, 2, 2, tol=1e-9):
        U = np.zeros((4, 8), dtype=float)
        V = np.zeros((4, 8), dtype=float)
        W = np.zeros((4, 8), dtype=float)
        c = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[i*2+j, c] = 1
                    V[j*2+k, c] = 1
                    W[k*2+i, c] = 1
                    c += 1
    return U, V, W

def _schoolbook_block(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W, rank

def _build_3x2x2():
    """
    3x2x2 decomposition.
    i=0,1: 2x2x2 with rank 7.
    i=2: 1x2x2 with rank 2.
    Total rank: 9.
    U shape: (6, 9), V shape: (4, 9), W shape: (6, 9)
    """
    n, m, p = 3, 2, 2
    U222, V222, W222 = _strassen_2x2x2()
    r222 = 7
    r122 = 2
    r_block = r222 + r122  # 9
    
    U = np.zeros((n * m, r_block), dtype=float)
    V = np.zeros((m * p, r_block), dtype=float)
    W = np.zeros((p * n, r_block), dtype=float)
    
    # Embed 2x2x2 part (i=0,1)
    for i in range(2):
        for j in range(2):
            for c in range(r222):
                U[i * m + j, c] = U222[i * 2 + j, c]
    for j in range(2):
        for k in range(2):
            for c in range(r222):
                V[j * p + k, c] = V222[j * 2 + k, c]
    for k in range(2):
        for i in range(2):
            for c in range(r222):
                W[k * n + i, c] = W222[k * 2 + i, c]
    
    # Embed 1x2x2 part (i=2)
    for j in range(2):
        for k in range(2):
            c_local = j * 2 + k
            c_global = r222 + c_local
            U[2 * m + j, c_global] = 1
            V[j * p + k, c_global] = 1
            W[k * n + 2, c_global] = 1
    
    return U, V, W, r_block

def _build_3x1x2():
    """
    3x1x2 decomposition. Rank 3.
    U shape: (3, 3), V shape: (2, 3), W shape: (6, 3)
    """
    n, m, p = 3, 1, 2
    rank = 3
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    c = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 0, c] = 1
            V[0 * p + k, c] = 1
            W[k * n + i, c] = 1
            c += 1
    return U, V, W, rank

def _build_3x3x2():
    """
    3x3x2 decomposition.
    j=0,1: 3x2x2 with rank 9.
    j=2: 3x1x2 with rank 3.
    Total rank: 12.
    U shape: (9, 12), V shape: (6, 12), W shape: (6, 12)
    """
    n, m, p = 3, 3, 2
    U322, V322, W322, r322 = _build_3x2x2()
    U312, V312, W312, r312 = _build_3x1x2()
    r_block = r322 + r312  # 12
    
    U = np.zeros((n * m, r_block), dtype=float)
    V = np.zeros((m * p, r_block), dtype=float)
    W = np.zeros((p * n, r_block), dtype=float)
    
    # Embed 3x2x2 part (j=0,1)
    for i in range(n):
        for j in range(2):
            for c in range(r322):
                U[i * m + j, c] = U322[i * 2 + j, c]
    for j in range(2):
        for k in range(p):
            for c in range(r322):
                V[j * p + k, c] = V322[j * 2 + k, c]
    for k in range(p):
        for i in range(n):
            for c in range(r322):
                W[k * n + i, c] = W322[k * n + i, c]
    
    # Embed 3x1x2 part (j=2)
    offset = r322
    for i in range(n):
        for c in range(r312):
            U[i * m + 2, offset + c] = U312[i * 1 + 0, c]
    for k in range(p):
        for c in range(r312):
            V[2 * p + k, offset + c] = V312[0 * 2 + k, c]
    for k in range(p):
        for i in range(n):
            for c in range(r312):
                W[k * n + i, offset + c] = W312[k * 3 + i, c]
    
    return U, V, W, r_block

def _build_3x3x6():
    """
    3x3x6 decomposition.
    Split p into three 2s.
    Each 3x3x2 block has rank 12.
    Total rank: 36.
    U shape: (9, 36), V shape: (18, 36), W shape: (18, 36)
    """
    n, m, p = 3, 3, 6
    U332, V332, W332, r332 = _build_3x3x2()
    total_rank = 3 * r332  # 36
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    for block_idx in range(3):
        p_offset = block_idx * 2
        offset = block_idx * r332
        
        for i in range(n):
            for j in range(m):
                for c in range(r332):
                    U[i * m + j, offset + c] = U332[i * m + j, c]
        for j in range(m):
            for k in range(2):
                for c in range(r332):
                    V[j * p + (p_offset + k), offset + c] = V332[j * 2 + k, c]
        for k in range(2):
            for i in range(n):
                for c in range(r332):
                    W[(p_offset + k) * n + i, offset + c] = W332[k * n + i, c]
    
    return U, V, W, total_rank

def _build_3x2x6():
    """
    3x2x6 decomposition.
    Split p into three 2s.
    Each 3x2x2 block has rank 9.
    Total rank: 27.
    U shape: (6, 27), V shape: (12, 27), W shape: (18, 27)
    """
    n, m, p = 3, 2, 6
    U322, V322, W322, r322 = _build_3x2x2()
    total_rank = 3 * r322  # 27
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    for block_idx in range(3):
        p_offset = block_idx * 2
        offset = block_idx * r322
        
        for i in range(n):
            for j in range(m):
                for c in range(r322):
                    U[i * m + j, offset + c] = U322[i * m + j, c]
        for j in range(m):
            for k in range(2):
                for c in range(r322):
                    V[j * p + (p_offset + k), offset + c] = V322[j * 2 + k, c]
        for k in range(2):
            for i in range(n):
                for c in range(r322):
                    W[(p_offset + k) * n + i, offset + c] = W322[k * n + i, c]
    
    return U, V, W, total_rank

def _build_decomposition(n, m, p):
    """
    Build 3x5x6 decomposition.
    Split m into 2 and 3.
    3x2x6: rank 27.
    3x3x6: rank 36.
    Total rank: 63.
    """
    m1, m2 = 2, 3
    
    U1, V1, W1, r1 = _build_3x2x6()
    U2, V2, W2, r2 = _build_3x3x6()
    
    total_rank = r1 + r2
    
    U_full = np.zeros((n * m, total_rank), dtype=float)
    V_full = np.zeros((m * p, total_rank), dtype=float)
    W_full = np.zeros((p * n, total_rank), dtype=float)
    
    # Embed U1 (3x2x6) into j=0,1
    for i in range(n):
        for j in range(m1):
            for c in range(r1):
                U_full[i * m + j, c] = U1[i * m1 + j, c]
    for j in range(m1):
        for k in range(p):
            for c in range(r1):
                V_full[j * p + k, c] = V1[j * p + k, c]
    for k in range(p):
        for i in range(n):
            for c in range(r1):
                W_full[k * n + i, c] = W1[k * n + i, c]
                
    # Embed U2 (3x3x6) into j=2,3,4
    offset = r1
    for i in range(n):
        for j in range(m2):
            for c in range(r2):
                U_full[i * m + (m1 + j), offset + c] = U2[i * m2 + j, c]
    for j in range(m2):
        for k in range(p):
            for c in range(r2):
                V_full[(m1 + j) * p + k, offset + c] = V2[j * p + k, c]
    for k in range(p):
        for i in range(n):
            for c in range(r2):
                W_full[k * n + i, offset + c] = W2[k * n + i, c]
                
    return U_full, V_full, W_full, total_rank

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s kill

    try:
        U, V, W, rank = _build_decomposition(n, m, p)
        if time.time() < deadline:
            if _check_exact(U, V, W, n, m, p, tol=1e-6):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
    
    # Fallback schoolbook
    U, V, W, rank = _schoolbook_block(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
