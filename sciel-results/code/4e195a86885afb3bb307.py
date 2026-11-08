import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _get_strassen_222():
    """
    Returns U, V, W for 2x2x2 matrix multiplication (Strassen).
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    """
    r = 7
    U = [[0]*r for _ in range(4)]
    V = [[0]*r for _ in range(4)]
    W = [[0]*r for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1 # C00 += M1, C11 += M1
    
    # M2 = (A10 + A11) B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1 # C10 += M2, C11 -= M2
    
    # M3 = A00 (B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1 # C01 += M3
    
    # M4 = A11 (B10 - B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1 # C00 += M4, C10 += M4
    
    # M5 = (A00 + A01) B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1 # C00 -= M5, C01 += M5
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1 # C11 += M6
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1 # C00 += M7
    
    return U, V, W

def _get_schoolbook(n, m, p):
    """Returns U, V, W for schoolbook multiplication."""
    r = n * m * p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decs, n, m, p, split_axis, split_indices):
    """
    Combines multiple decompositions into one.
    decs: list of (U, V, W, n_sub, m_sub, p_sub, offset_i, offset_j, offset_k)
    """
    total_rank = sum(len(U[0]) for U, V, W, *_ in decs)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    rank_offset = 0
    for U_sub, V_sub, W_sub, n_sub, m_sub, p_sub, off_i, off_j, off_k in decs:
        r_sub = len(U_sub[0])
        # Map U_sub (n_sub * m_sub) to U (n * m)
        for i_sub in range(n_sub):
            for j_sub in range(m_sub):
                i_global = off_i + i_sub
                j_global = off_j + j_sub
                for c in range(r_sub):
                    U[i_global*m + j_global][rank_offset + c] = U_sub[i_sub*m_sub + j_sub][c]
        
        # Map V_sub (m_sub * p_sub) to V (m * p)
        for j_sub in range(m_sub):
            for k_sub in range(p_sub):
                j_global = off_j + j_sub
                k_global = off_k + k_sub
                for c in range(r_sub):
                    V[j_global*p + k_global][rank_offset + c] = V_sub[j_sub*p_sub + k_sub][c]
        
        # Map W_sub (p_sub * n_sub) to W (p * n)
        for k_sub in range(p_sub):
            for i_sub in range(n_sub):
                k_global = off_k + k_sub
                i_global = off_i + i_sub
                for c in range(r_sub):
                    W[k_global*n + i_global][rank_offset + c] = W_sub[k_sub*n_sub + i_sub][c]
        
        rank_offset += r_sub
    
    return U, V, W

def _get_decomposition(n, m, p, depth=0):
    """
    Recursively finds a decomposition for n x m x p.
    """
    # Base cases
    if n == 2 and m == 2 and p == 2:
        return _get_strassen_222()
    
    if n == 1 or m == 1 or p == 1:
        return _get_schoolbook(n, m, p)
    
    # Try splitting m
    if m >= 2:
        # Try split 2 + rest or 3 + rest if possible
        splits = []
        if m >= 2: splits.append(2)
        if m >= 3: splits.append(3)
        
        for m1 in splits:
            m2 = m - m1
            if m2 < 1: continue
            
            dec1 = _get_decomposition(n, m1, p, depth+1)
            dec2 = _get_decomposition(n, m2, p, depth+1)
            
            rank1 = len(dec1[0][0])
            rank2 = len(dec2[0][0])
            
            # Combine
            U, V, W = _combine_decompositions(
                [
                    (dec1[0], dec1[1], dec1[2], n, m1, p, 0, 0, 0),
                    (dec2[0], dec2[1], dec2[2], n, m2, p, 0, m1, 0)
                ],
                n, m, p
            )
            return U, V, W

    # Try splitting p
    if p >= 2:
        splits = []
        if p >= 2: splits.append(2)
        if p >= 3: splits.append(3)
        
        for p1 in splits:
            p2 = p - p1
            if p2 < 1: continue
            
            dec1 = _get_decomposition(n, m, p1, depth+1)
            dec2 = _get_decomposition(n, m, p2, depth+1)
            
            U, V, W = _combine_decompositions(
                [
                    (dec1[0], dec1[1], dec1[2], n, m, p1, 0, 0, 0),
                    (dec2[0], dec2[1], dec2[2], n, m, p2, 0, 0, p1)
                ],
                n, m, p
            )
            return U, V, W

    # Try splitting n
    if n >= 2:
        n1 = 1
        n2 = n - 1
        dec1 = _get_decomposition(n1, m, p, depth+1)
        dec2 = _get_decomposition(n2, m, p, depth+1)
        
        U, V, W = _combine_decompositions(
            [
                (dec1[0], dec1[1], dec1[2], n1, m, p, 0, 0, 0),
                (dec2[0], dec2[1], dec2[2], n2, m, p, n1, 0, 0)
            ],
            n, m, p
        )
        return U, V, W

    # Fallback to schoolbook
    return _get_schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    U, V, W = _get_decomposition(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        U, V, W = _get_schoolbook(n, m, p)
        
    return {"U": U, "V": V, "W": W}
