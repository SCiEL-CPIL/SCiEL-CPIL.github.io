import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Try different partitions
    best_rank = None
    best_partition = None
    best_result = None
    
    # Candidate partitions
    partitions = [
        [2, 2, 1],
        [1, 2, 2],
        [2, 1, 2],
        [1, 1, 3],
        [3, 1, 1],
        [1, 3, 1],
    ]
    
    for n_parts in partitions:
        for m_parts in partitions:
            for p_parts in partitions:
                if sum(n_parts) != n or sum(m_parts) != m or sum(p_parts) != p:
                    continue
                
                # Calculate rank for this partition
                rank = 0
                for i in range(len(n_parts)):
                    for j in range(len(m_parts)):
                        for k in range(len(p_parts)):
                            ni = n_parts[i]
                            mj = m_parts[j]
                            pk = p_parts[k]
                            if ni == 2 and mj == 2 and pk == 2:
                                rank += 7
                            else:
                                rank += ni * mj * pk
                
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_partition = (n_parts, m_parts, p_parts)
    
    n_parts, m_parts, p_parts = best_partition
    
    # Build the decomposition
    n_starts = [0]
    m_starts = [0]
    p_starts = [0]
    for i in range(1, len(n_parts)):
        n_starts.append(n_starts[-1] + n_parts[i-1])
    for i in range(1, len(m_parts)):
        m_starts.append(m_starts[-1] + m_parts[i-1])
    for i in range(1, len(p_parts)):
        p_starts.append(p_starts[-1] + p_parts[i-1])
    
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    
    col = 0
    
    for i in range(len(n_parts)):
        for j in range(len(m_parts)):
            for k in range(len(p_parts)):
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                
                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen for 2x2x2
                    i0, i1 = i_start, i_start + 1
                    j0, j1 = j_start, j_start + 1
                    k0, k1 = k_start, k_start + 1
                    
                    # M1 = (A11+A22)(B11+B22)
                    U[i0 * m + j0][col] = 1
                    U[i1 * m + j1][col] = 1
                    V[j0 * p + k0][col] = 1
                    V[j1 * p + k1][col] = 1
                    W[k0 * n + i0][col] = 1
                    W[k1 * n + i1][col] = 1
                    col += 1
                    
                    # M2 = (A21+A22)B11
                    U[i1 * m + j0][col] = 1
                    U[i1 * m + j1][col] = 1
                    V[j0 * p + k0][col] = 1
                    W[k0 * n + i1][col] = 1
                    W[k1 * n + i1][col] = -1
                    col += 1
                    
                    # M3 = A11(B21-B22)
                    U[i0 * m + j0][col] = 1
                    V[j0 * p + k1][col] = 1
                    V[j1 * p + k1][col] = -1
                    W[k1 * n + i0][col] = 1
                    W[k1 * n + i1][col] = 1
                    col += 1
                    
                    # M4 = A22(B11-B12)
                    U[i1 * m + j1][col] = 1
                    V[j1 * p + k0][col] = 1
                    V[j0 * p + k0][col] = -1
                    W[k0 * n + i0][col] = 1
                    W[k0 * n + i1][col] = 1
                    col += 1
                    
                    # M5 = (A11+A12)B22
                    U[i0 * m + j0][col] = 1
                    U[i0 * m + j1][col] = 1
                    V[j1 * p + k1][col] = 1
                    W[k0 * n + i0][col] = -1
                    W[k1 * n + i0][col] = 1
                    col += 1
                    
                    # M6 = (A21-A11)(B11+B12)
                    U[i1 * m + j0][col] = 1
                    U[i0 * m + j0][col] = -1
                    V[j0 * p + k0][col] = 1
                    V[j0 * p + k1][col] = 1
                    W[k1 * n + i1][col] = 1
                    col += 1
                    
                    # M7 = (A12-A22)(B21+B22)
                    U[i0 * m + j1][col] = 1
                    U[i1 * m + j1][col] = -1
                    V[j1 * p + k0][col] = 1
                    V[j1 * p + k1][col] = 1
                    W[k0 * n + i0][col] = 1
                    col += 1
                    
                else:
                    # Schoolbook for other blocks
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                U[(i_start + u) * m + (j_start + v)][col] = 1
                                V[(j_start + v) * p + (k_start + w)][col] = 1
                                W[(k_start + w) * n + (i_start + u)][col] = 1
                                col += 1
    
    return {"U": U, "V": V, "W": W}
