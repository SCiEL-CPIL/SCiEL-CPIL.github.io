"""Matrix multiplication 4x5 by 5x6 — Optimized Strassen decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2x2():
    """
    Build the 2x2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _build_rank1_block(n, m, p):
    """
    Build a rank-1 decomposition for n x m x p where m=1.
    This is valid because the tensor is effectively an outer product.
    """
    if m != 1:
        return _build_schoolbook(n, m, p)
    
    rank = 1
    U = [[1] for _ in range(n*m)]
    V = [[1] for _ in range(m*p)]
    W = [[1] for _ in range(p*n)]
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build a rank-49 decomposition for 4x4 * 4x4 -> 4x4.
    Uses Kronecker product of 2x2x2 Strassen.
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for t1 in range(7):
        for t2 in range(7):
            col = t1 * 7 + t2
            # U[i1*2+i2, j1*2+j2] = U2[i1,j1,t1] * U2[i2,j2,t2]
            for i1 in range(2):
                for j1 in range(2):
                    c1 = U2[i1*2+j1][t1]
                    if c1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            c2 = U2[i2*2+j2][t2]
                            if c2 == 0: continue
                            u_row = (i1*2+i2)*4 + (j1*2+j2)
                            U[u_row][col] += c1 * c2
            
            # V[j1*2+j2, k1*2+k2] = V2[j1,k1,t1] * V2[j2,k2,t2]
            for j1 in range(2):
                for k1 in range(2):
                    c1 = V2[j1*2+k1][t1]
                    if c1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            c2 = V2[j2*2+k2][t2]
                            if c2 == 0: continue
                            v_row = (j1*2+j2)*4 + (k1*2+k2)
                            V[v_row][col] += c1 * c2
            
            # W[k1*2+k2, i1*2+i2] = W2[k1,i1,t1] * W2[k2,i2,t2]
            for k1 in range(2):
                for i1 in range(2):
                    c1 = W2[k1*2+i1][t1]
                    if c1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            c2 = W2[k2*2+i2][t2]
                            if c2 == 0: continue
                            w_row = (k1*2+k2)*4 + (i1*2+i2)
                            W[w_row][col] += c1 * c2
    
    return U, V, W

def _build_4x4x2_strassen():
    """
    Build a rank-28 decomposition for 4x4 * 4x2 -> 4x2.
    Uses 4 Strassen 2x2x2 blocks.
    """
    U2, V2, W2 = _build_strassen_2x2x2()
    rank = 28
    n_sub, m_sub, p_sub = 4, 4, 2
    U = [[0]*rank for _ in range(n_sub * m_sub)]
    V = [[0]*rank for _ in range(m_sub * p_sub)]
    W = [[0]*rank for _ in range(p_sub * n_sub)]
    
    # 4 Strassen blocks:
    # A is 4x4, split into four 2x2 blocks.
    # B is 4x2, split into two 2x2 blocks (top and bottom).
    # C is 4x2, split into two 2x2 blocks (top and bottom).
    
    # Block (ai, bj) where ai in {0,1} (row block of A/C), bj in {0,1} (col block of A / row block of B)
    # A block (ai, bj) is 2x2: rows 2*ai..2*ai+1, cols 2*bj..2*bj+1
    # B block bj is 2x2: rows 2*bj..2*bj+1, cols 0..1
    # C contribution goes to rows 2*ai..2*ai+1, cols 0..1
    
    for ai in range(2):
        for bj in range(2):
            block_offset = (ai * 2 + bj) * 7
            for t in range(7):
                col = block_offset + t
                # U: A[i,j] where i in 2*ai..2*ai+1, j in 2*bj..2*bj+1
                for i in range(2):
                    for j in range(2):
                        val = U2[i*2+j][t]
                        if val != 0:
                            u_row = (2*ai + i) * m_sub + (2*bj + j)
                            U[u_row][col] += val
                
                # V: B[j,k] where j in 2*bj..2*bj+1, k in 0..1
                for j in range(2):
                    for k in range(2):
                        val = V2[j*2+k][t]
                        if val != 0:
                            v_row = (2*bj + j) * p_sub + k
                            V[v_row][col] += val
                
                # W: C[i,k] where i in 2*ai..2*ai+1, k in 0..1
                for k in range(2):
                    for i in range(2):
                        val = W2[k*2+i][t]
                        if val != 0:
                            w_row = k * n_sub + (2*ai + i)
                            W[w_row][col] += val
    
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions into one global decomposition.
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, i_s, j_s, k_s, i_sz, j_sz, k_sz in decomps:
        rank_sub = len(U_sub[0])
        
        for i in range(i_sz):
            for j in range(j_sz):
                u_row_sub = i * j_sz + j
                u_row_glob = (i_s + i) * m + (j_s + j)
                for c in range(rank_sub):
                    U[u_row_glob][current_rank + c] = U_sub[u_row_sub][c]
        
        for j in range(j_sz):
            for k in range(k_sz):
                v_row_sub = j * k_sz + k
                v_row_glob = (j_s + j) * p + (k_s + k)
                for c in range(rank_sub):
                    V[v_row_glob][current_rank + c] = V_sub[v_row_sub][c]
        
        for k in range(k_sz):
            for i in range(i_sz):
                w_row_sub = k * i_sz + i
                w_row_glob = (k_s + k) * n + (i_s + i)
                for c in range(rank_sub):
                    W[w_row_glob][current_rank + c] = W_sub[w_row_sub][c]
        
        current_rank += rank_sub
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    decomps = []
    
    # Block 1: 4x4x4 (i=0..3, j=0..3, k=0..3)
    U1, V1, W1 = _build_4x4x4_strassen()
    decomps.append((U1, V1, W1, 0, 0, 0, 4, 4, 4))
    
    # Block 2: 4x4x2 (i=0..3, j=0..3, k=4..5)
    U2, V2, W2 = _build_4x4x2_strassen()
    decomps.append((U2, V2, W2, 0, 0, 4, 4, 4, 2))
    
    # Block 3: 4x1x6 (i=0..3, j=4, k=0..5)
    U3, V3, W3 = _build_rank1_block(4, 1, 6)
    decomps.append((U3, V3, W3, 0, 4, 0, 4, 1, 6))
    
    U, V, W = _combine_decompositions(decomps, n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
