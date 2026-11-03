"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _strassen_factors(n, m, p):
    """
    Recursive Strassen decomposition for n x m x p matrix multiplication.
    Returns U (n*m x rank), V (m*p x rank), W (p*n x rank) as lists.
    
    W indexing: W[k*n + i, c] is coefficient for C[i, k].
    """
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    
    if p == 1:
        rank = n * m
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m)]
        W = [[0]*rank for _ in range(n)]
        col = 0
        for i in range(n):
            for j in range(m):
                U[i*m+j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, W
    
    if n != m or m != p:
        rank = n * m * p
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
        return U, V, W
    
    # Strassen for n x n x n
    n2 = n // 2
    
    terms = [
        {'A': [(0,1),(3,1)], 'B': [(0,1),(3,1)], 'C': [(0,1),(3,1)]},
        {'A': [(2,1),(3,1)], 'B': [(0,1)], 'C': [(2,1)]},
        {'A': [(0,1)], 'B': [(1,1),(3,-1)], 'C': [(1,1)]},
        {'A': [(3,1)], 'B': [(2,1),(0,-1)], 'C': [(0,1),(2,1)]},
        {'A': [(0,1),(1,1)], 'B': [(3,1)], 'C': [(0,-1),(1,1)]},
        {'A': [(2,1),(0,-1)], 'B': [(0,1),(1,1)], 'C': [(3,1)]},
        {'A': [(1,1),(3,-1)], 'B': [(2,1),(3,1)], 'C': [(0,1),(3,-1)]},
    ]
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    for term in terms:
        U_sub, V_sub, W_sub = _strassen_factors(n2, n2, n2)
        rank_sub = len(U_sub[0])
        
        U_full = np.zeros((n*n, rank_sub))
        V_full = np.zeros((n*n, rank_sub))
        W_full = np.zeros((n*n, rank_sub))
        
        for (a_block, a_coeff) in term['A']:
            for i2 in range(n2):
                for j2 in range(n2):
                    if a_block == 0:
                        row = i2 * n + j2
                    elif a_block == 1:
                        row = i2 * n + (n2 + j2)
                    elif a_block == 2:
                        row = (n2 + i2) * n + j2
                    else:
                        row = (n2 + i2) * n + (n2 + j2)
                    U_full[row, :] = a_coeff * np.array(U_sub[i2*n2 + j2], dtype=float)
        
        for (b_block, b_coeff) in term['B']:
            for i2 in range(n2):
                for j2 in range(n2):
                    if b_block == 0:
                        row = i2 * n + j2
                    elif b_block == 1:
                        row = i2 * n + (n2 + j2)
                    elif b_block == 2:
                        row = (n2 + i2) * n + j2
                    else:
                        row = (n2 + i2) * n + (n2 + j2)
                    V_full[row, :] = b_coeff * np.array(V_sub[i2*n2 + j2], dtype=float)
        
        for (c_block, c_coeff) in term['C']:
            for i2 in range(n2):
                for j2 in range(n2):
                    if c_block == 0:
                        row = j2 * n + i2
                    elif c_block == 1:
                        row = j2 * n + (n2 + i2)
                    elif c_block == 2:
                        row = (n2 + j2) * n + i2
                    else:
                        row = (n2 + j2) * n + (n2 + i2)
                    W_full[row, :] = c_coeff * np.array(W_sub[j2*n2 + i2], dtype=float)
        
        U_parts.append(U_full)
        V_parts.append(V_full)
        W_parts.append(W_full)
    
    U_final = np.hstack(U_parts)
    V_final = np.hstack(V_parts)
    W_final = np.hstack(W_parts)
    
    return U_final.tolist(), V_final.tolist(), W_final.tolist()


def _schoolbook_factors(n, m, p):
    rank = n * m * p
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
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    p1, p2, p3 = 4, 4, 1
    
    U1, V1, W1 = _strassen_factors(n, m, p1)
    U2, V2, W2 = _strassen_factors(n, m, p2)
    U3, V3, W3 = _schoolbook_factors(n, m, p3)
    
    r1, r2, r3 = len(U1[0]), len(U2[0]), len(U3[0])
    total_rank = r1 + r2 + r3
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Sub 1: B cols 0-3, C cols 0-3
    U[:, :r1] = np.array(U1, dtype=float)
    for j in range(m):
        for k in range(p1):
            V[j*p + k, :r1] = V1[j*p1 + k]
    for k in range(p1):
        for i in range(n):
            W[k*n + i, :r1] = W1[k*p1 + i]
    
    # Sub 2: B cols 4-7, C cols 4-7
    U[:, r1:r1+r2] = np.array(U2, dtype=float)
    for j in range(m):
        for k in range(p2):
            V[j*p + p1 + k, r1:r1+r2] = V2[j*p2 + k]
    for k in range(p2):
        for i in range(n):
            W[(p1+k)*n + i, r1:r1+r2] = W2[k*p2 + i]
    
    # Sub 3: B col 8, C col 8
    U[:, r1+r2:] = np.array(U3, dtype=float)
    for j in range(m):
        V[j*p + 8, r1+r2:] = V3[j]
    for i in range(n):
        W[8*n + i, r1+r2:] = W3[i]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

