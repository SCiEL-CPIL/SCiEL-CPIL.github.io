import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    n, m, p = 3, 5, 5
    
    # Block partition:
    # n = 1 + 2
    # m = 2 + 1 + 2
    # p = 2 + 1 + 2
    
    # Ranks:
    # (1,2,2): 4
    # (1,1,2): 2
    # (1,2,1): 2
    # (1,1,1): 1
    # (2,2,2): 7 (Strassen)
    # (2,1,2): 4
    # (2,2,1): 4
    # (2,1,1): 2
    
    # Total:
    # i=0 (n=1):
    #   j=0 (m=2), k=0 (p=2): (1,2,2) -> 4
    #   j=0 (m=2), k=1 (p=1): (1,2,1) -> 2
    #   j=0 (m=2), k=2 (p=2): (1,2,2) -> 4
    #   j=1 (m=1), k=0 (p=2): (1,1,2) -> 2
    #   j=1 (m=1), k=1 (p=1): (1,1,1) -> 1
    #   j=1 (m=1), k=2 (p=2): (1,1,2) -> 2
    #   j=2 (m=2), k=0 (p=2): (1,2,2) -> 4
    #   j=2 (m=2), k=1 (p=1): (1,2,1) -> 2
    #   j=2 (m=2), k=2 (p=2): (1,2,2) -> 4
    #   Sum i=0: 4+2+4+2+1+2+4+2+4 = 25
    
    # i=1 (n=2):
    #   j=0 (m=2), k=0 (p=2): (2,2,2) -> 7
    #   j=0 (m=2), k=1 (p=1): (2,2,1) -> 4
    #   j=0 (m=2), k=2 (p=2): (2,2,2) -> 7
    #   j=1 (m=1), k=0 (p=2): (2,1,2) -> 4
    #   j=1 (m=1), k=1 (p=1): (2,1,1) -> 2
    #   j=1 (m=1), k=2 (p=2): (2,1,2) -> 4
    #   j=2 (m=2), k=0 (p=2): (2,2,2) -> 7
    #   j=2 (m=2), k=1 (p=1): (2,2,1) -> 4
    #   j=2 (m=2), k=2 (p=2): (2,2,2) -> 7
    #   Sum i=1: 7+4+7+4+2+4+7+4+7 = 46
    
    # Total Rank = 25 + 46 = 71

    total_rank = 71
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    n_parts = [1, 2]
    m_parts = [2, 1, 2]
    p_parts = [2, 1, 2]
    n_off = [0, 1]
    m_off = [0, 2, 3]
    p_off = [0, 2, 3]
    
    def add_schoolbook(i0, j0, k0, ns, ms, ps, U, V, W, col):
        for i_s in range(ns):
            for j_s in range(ms):
                for k_s in range(ps):
                    i = i0 + i_s
                    j = j0 + j_s
                    k = k0 + k_s
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col

    def add_strassen(i0, j0, k0, U, V, W, col):
        # Strassen 2x2x2
        # M1 = (A00+A11)*(B00+B11)
        U[i0*m+j0][col] = 1; U[(i0+1)*m+(j0+1)][col] = 1
        V[j0*p+k0][col] = 1; V[(j0+1)*p+(k0+1)][col] = 1
        W[k0*n+i0][col] = 1; W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        
        # M2 = (A10+A11)*B00
        U[(i0+1)*m+j0][col] = 1; U[(i0+1)*m+(j0+1)][col] = 1
        V[j0*p+k0][col] = 1
        W[k0*n+(i0+1)][col] = 1; W[(k0+1)*n+(i0+1)][col] = -1
        col += 1
        
        # M3 = A00*(B01-B11)
        U[i0*m+j0][col] = 1
        V[j0*p+(k0+1)][col] = 1; V[(j0+1)*p+(k0+1)][col] = -1
        W[(k0+1)*n+i0][col] = 1; W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        
        # M4 = A11*(B10-B00)
        U[(i0+1)*m+(j0+1)][col] = 1
        V[(j0+1)*p+k0][col] = 1; V[j0*p+k0][col] = -1
        W[k0*n+i0][col] = 1; W[k0*n+(i0+1)][col] = 1
        col += 1
        
        # M5 = (A00+A01)*B11
        U[i0*m+j0][col] = 1; U[i0*m+(j0+1)][col] = 1
        V[(j0+1)*p+(k0+1)][col] = 1
        W[k0*n+i0][col] = -1; W[(k0+1)*n+i0][col] = 1
        col += 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[(i0+1)*m+j0][col] = 1; U[i0*m+j0][col] = -1
        V[j0*p+k0][col] = 1; V[j0*p+(k0+1)][col] = 1
        W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[i0*m+(j0+1)][col] = 1; U[(i0+1)*m+(j0+1)][col] = -1
        V[(j0+1)*p+k0][col] = 1; V[(j0+1)*p+(k0+1)][col] = 1
        W[k0*n+i0][col] = 1
        col += 1
        
        return col

    for bi in range(2):
        for bj in range(3):
            for bk in range(3):
                ns = n_parts[bi]
                ms = m_parts[bj]
                ps = p_parts[bk]
                i0 = n_off[bi]
                j0 = m_off[bj]
                k0 = p_off[bk]
                
                if ns == 2 and ms == 2 and ps == 2:
                    col = add_strassen(i0, j0, k0, U, V, W, col)
                else:
                    col = add_schoolbook(i0, j0, k0, ns, ms, ps, U, V, W, col)
                    
    if col < total_rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        
    return {"U": U, "V": V, "W": W}
