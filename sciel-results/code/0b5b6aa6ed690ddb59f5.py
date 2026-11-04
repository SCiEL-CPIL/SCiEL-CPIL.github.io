import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Partitions
    # n: 2, 1
    # m: 2, 2
    # p: 2, 2, 2, 2
    n_parts = [2, 1]
    m_parts = [2, 2]
    p_parts = [2, 2, 2, 2]
    
    # Calculate offsets
    n_offsets = [0]
    for x in n_parts[:-1]:
        n_offsets.append(n_offsets[-1] + x)
        
    m_offsets = [0]
    for x in m_parts[:-1]:
        m_offsets.append(m_offsets[-1] + x)
        
    p_offsets = [0]
    for x in p_parts[:-1]:
        p_offsets.append(p_offsets[-1] + x)

    # Initialize U, V, W with zeros
    # We don't know the final rank yet, so we'll collect lists of vectors
    # Or we can estimate max rank and pad.
    # Max rank per block:
    # 2x2x2 -> 7
    # 2x2x2 -> 7
    # 1x2x2 -> 4
    # 1x2x2 -> 4
    # Total 22 blocks? No.
    # s in 0..1, t in 0..1, u in 0..3
    # 2 * 2 * 4 = 16 blocks.
    # 8 blocks of 2x2x2 (Rank 7) -> 56
    # 8 blocks of 1x2x2 (Rank 4) -> 32
    # Total Rank = 88.
    
    max_rank = 100
    U = [[0.0] * max_rank for _ in range(n * m)]
    V = [[0.0] * max_rank for _ in range(m * p)]
    W = [[0.0] * max_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Strassen coefficients for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Local indices for 2x2:
    # A: 0,1,2,3 (i*m+j) -> (0,0),(0,1),(1,0),(1,1)
    # B: 0,1,2,3 (j*p+k) -> (0,0),(0,1),(1,0),(1,1)
    # C: 0,1,2,3 (k*n+i) -> (0,0),(0,1),(1,0),(1,1)
    
    def add_strassen_block(U, V, W, i_off, j_off, k_off, rank_start):
        # r=0: M1
        # U: a, d -> (0,0), (1,1)
        U[i_off*m + j_off + 0][rank_start] = 1
        U[i_off*m + j_off + 3][rank_start] = 1
        # V: e, h -> (0,0), (1,1)
        V[j_off*p + k_off + 0][rank_start] = 1
        V[j_off*p + k_off + 3][rank_start] = 1
        # W: C11, C22 -> (0,0), (1,1)
        W[k_off*n + i_off + 0][rank_start] = 1
        W[k_off*n + i_off + 3][rank_start] = 1
        
        # r=1: M2
        # U: c, d -> (1,0), (1,1)
        U[i_off*m + j_off + 2][rank_start+1] = 1
        U[i_off*m + j_off + 3][rank_start+1] = 1
        # V: e -> (0,0)
        V[j_off*p + k_off + 0][rank_start+1] = 1
        # W: C21 -> (1,0)
        W[k_off*n + i_off + 2][rank_start+1] = 1
        
        # r=2: M3
        # U: a -> (0,0)
        U[i_off*m + j_off + 0][rank_start+2] = 1
        # V: f, -h -> (0,1), (1,1)
        V[j_off*p + k_off + 1][rank_start+2] = 1
        V[j_off*p + k_off + 3][rank_start+2] = -1
        # W: C12, C22 -> (0,1), (1,1)
        W[k_off*n + i_off + 1][rank_start+2] = 1
        W[k_off*n + i_off + 3][rank_start+2] = 1
        
        # r=3: M4
        # U: d -> (1,1)
        U[i_off*m + j_off + 3][rank_start+3] = 1
        # V: g, -e -> (1,0), (0,0)
        V[j_off*p + k_off + 2][rank_start+3] = 1
        V[j_off*p + k_off + 0][rank_start+3] = -1
        # W: C11, C21 -> (0,0), (1,0)
        W[k_off*n + i_off + 0][rank_start+3] = 1
        W[k_off*n + i_off + 2][rank_start+3] = 1
        
        # r=4: M5
        # U: a, b -> (0,0), (0,1)
        U[i_off*m + j_off + 0][rank_start+4] = 1
        U[i_off*m + j_off + 1][rank_start+4] = 1
        # V: h -> (1,1)
        V[j_off*p + k_off + 3][rank_start+4] = 1
        # W: -C11, C12 -> (0,0), (0,1)
        W[k_off*n + i_off + 0][rank_start+4] = -1
        W[k_off*n + i_off + 1][rank_start+4] = 1
        
        # r=5: M6
        # U: c, -a -> (1,0), (0,0)
        U[i_off*m + j_off + 2][rank_start+5] = 1
        U[i_off*m + j_off + 0][rank_start+5] = -1
        # V: e, f -> (0,0), (0,1)
        V[j_off*p + k_off + 0][rank_start+5] = 1
        V[j_off*p + k_off + 1][rank_start+5] = 1
        # W: C22 -> (1,1)
        W[k_off*n + i_off + 3][rank_start+5] = 1
        
        # r=6: M7
        # U: b, -d -> (0,1), (1,1)
        U[i_off*m + j_off + 1][rank_start+6] = 1
        U[i_off*m + j_off + 3][rank_start+6] = -1
        # V: g, h -> (1,0), (1,1)
        V[j_off*p + k_off + 2][rank_start+6] = 1
        V[j_off*p + k_off + 3][rank_start+6] = 1
        # W: C11 -> (0,0)
        W[k_off*n + i_off + 0][rank_start+6] = 1
        
        return rank_start + 7

    def add_schoolbook_block(U, V, W, i_off, j_off, k_off, n_s, m_s, p_s, rank_start):
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    r = rank_start
                    U[(i_off+i)*m + (j_off+j)][r] = 1
                    V[(j_off+j)*p + (k_off+k)][r] = 1
                    W[(k_off+k)*n + (i_off+i)][r] = 1
                    rank_start += 1
        return rank_start

    for s, n_s in enumerate(n_parts):
        for t, m_s in enumerate(m_parts):
            for u, p_s in enumerate(p_parts):
                i_off = n_offsets[s]
                j_off = m_offsets[t]
                k_off = p_offsets[u]
                
                if n_s == 2 and m_s == 2 and p_s == 2:
                    current_rank = add_strassen_block(U, V, W, i_off, j_off, k_off, current_rank)
                else:
                    current_rank = add_schoolbook_block(U, V, W, i_off, j_off, k_off, n_s, m_s, p_s, current_rank)
    
    # Trim matrices to current_rank
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
