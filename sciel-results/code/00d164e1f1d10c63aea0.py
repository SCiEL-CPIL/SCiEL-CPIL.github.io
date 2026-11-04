import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 0
    U = []
    V = []
    W = []
    
    # Partition sizes
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 1]
    
    # Calculate total rank
    # 4 blocks of 2x2x2 (Strassen rank 7)
    # Rest schoolbook
    rank = 4 * 7
    for i in range(len(n_splits)):
        for j in range(len(m_splits)):
            for k in range(len(p_splits)):
                ni = n_splits[i]
                mj = m_splits[j]
                pk = p_splits[k]
                if ni == 2 and mj == 2 and pk == 2:
                    continue # Already counted
                rank += ni * mj * pk
                
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Precompute offsets
    n_offsets = [0]
    for x in n_splits[:-1]:
        n_offsets.append(n_offsets[-1] + x)
    m_offsets = [0]
    for x in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + x)
    p_offsets = [0]
    for x in p_splits[:-1]:
        p_offsets.append(p_offsets[-1] + x)

    for i in range(len(n_splits)):
        for j in range(len(m_splits)):
            for k in range(len(p_splits)):
                ni = n_splits[i]
                mj = m_splits[j]
                pk = p_splits[k]
                
                i0 = n_offsets[i]
                j0 = m_offsets[j]
                k0 = p_offsets[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen
                    # u, v, w vectors for 7 terms
                    # Indices for A (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
                    # Indices for B (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
                    # Indices for C (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
                    
                    # Term 1: (A00+A11)(B00+B11) -> C00+C10
                    u1 = [1, 0, 0, 1]
                    v1 = [1, 0, 0, 1]
                    w1 = [1, 0, 1, 0]
                    
                    # Term 2: (A10+A11)B00 -> C10
                    u2 = [0, 0, 1, 1]
                    v2 = [1, 0, 0, 0]
                    w2 = [0, 0, 1, 0]
                    
                    # Term 3: A00(B01-B11) -> C01+C11
                    u3 = [1, 0, 0, 0]
                    v3 = [0, 1, 0, -1]
                    w3 = [0, 1, 0, 1]
                    
                    # Term 4: A11(B10-B00) -> -C00+C01
                    u4 = [0, 0, 0, 1]
                    v4 = [0, 0, 1, -1]
                    w4 = [-1, 1, 0, 0]
                    
                    # Term 5: (A00+A01)B11 -> C00+C11
                    u5 = [1, 1, 0, 0]
                    v5 = [0, 0, 0, 1]
                    w5 = [1, 0, 0, 1]
                    
                    # Term 6: (A10-A00)(B00+B01) -> C11
                    u6 = [-1, 0, 1, 0]
                    v6 = [1, 1, 0, 0]
                    w6 = [0, 0, 0, 1]
                    
                    # Term 7: (A01-A11)(B10+B11) -> -C00
                    u7 = [0, 1, 0, -1]
                    v7 = [0, 0, 1, 1]
                    w7 = [-1, 0, 0, 0]
                    
                    terms = [
                        (u1, v1, w1),
                        (u2, v2, w2),
                        (u3, v3, w3),
                        (u4, v4, w4),
                        (u5, v5, w5),
                        (u6, v6, w6),
                        (u7, v7, w7)
                    ]
                    
                    for u, v, w in terms:
                        # Map local indices to global
                        for lu in range(4):
                            lu_i = lu // 2
                            lu_j = lu % 2
                            gu = (i0 + lu_i) * m + (j0 + lu_j)
                            U[gu][col] = u[lu]
                            
                            gv = (j0 + lu_i) * p + (k0 + lu_j)
                            V[gv][col] = v[lu]
                            
                            gw = (k0 + lu_i) * n + (i0 + lu_j)
                            W[gw][col] = w[lu]
                        col += 1

                else:
                    # Schoolbook
                    for ui in range(ni):
                        for uj in range(mj):
                            for uk in range(pk):
                                gu = (i0 + ui) * m + (j0 + uj)
                                gv = (j0 + uj) * p + (k0 + uk)
                                gw = (k0 + uk) * n + (i0 + ui)
                                
                                U[gu][col] = 1
                                V[gv][col] = 1
                                W[gw][col] = 1
                                col += 1

    return {"U": U, "V": V, "W": W}
