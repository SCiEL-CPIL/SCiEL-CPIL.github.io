import time
import numpy as np

DIMENSIONS = [3, 5, 5]
_DEADLINE = time.time() + 140  # safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    rank = 71
    
    U = [[0] * rank for _ in range(n * m)]  # 15 x 71
    V = [[0] * rank for _ in range(m * p)]  # 25 x 71
    W = [[0] * rank for _ in range(p * n)]  # 15 x 71
    
    col = 0
    
    # Helper for schoolbook sub-block
    def schoolbook(i0, j0, k0, n_s, m_s, p_s, U, V, W, col):
        for i_s in range(n_s):
            for j_s in range(m_s):
                for k_s in range(p_s):
                    i = i0 + i_s
                    j = j0 + j_s
                    k = k0 + k_s
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col
    
    # Helper for Strassen 2x2x2 block
    def strassen2(i0, j0, k0, U, V, W, col):
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        U[(i0+0)*m+(j0+0)][col] = 1; U[(i0+1)*m+(j0+1)][col] = 1
        V[(j0+0)*p+(k0+0)][col] = 1; V[(j0+1)*p+(k0+1)][col] = 1
        W[(k0+0)*n+(i0+0)][col] = 1; W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        U[(i0+1)*m+(j0+0)][col] = 1; U[(i0+1)*m+(j0+1)][col] = 1
        V[(j0+0)*p+(k0+0)][col] = 1
        W[(k0+0)*n+(i0+1)][col] = 1; W[(k0+1)*n+(i0+1)][col] = -1
        col += 1
        # M3 = A[0,0] * (B[0,1]-B[1,1])
        U[(i0+0)*m+(j0+0)][col] = 1
        V[(j0+0)*p+(k0+1)][col] = 1; V[(j0+1)*p+(k0+1)][col] = -1
        W[(k0+1)*n+(i0+0)][col] = 1; W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        # M4 = A[1,1] * (B[1,0]-B[0,0])
        U[(i0+1)*m+(j0+1)][col] = 1
        V[(j0+1)*p+(k0+0)][col] = 1; V[(j0+0)*p+(k0+0)][col] = -1
        W[(k0+0)*n+(i0+0)][col] = 1; W[(k0+0)*n+(i0+1)][col] = 1
        col += 1
        # M5 = (A[0,0]+A[0,1]) * B[1,1]
        U[(i0+0)*m+(j0+0)][col] = 1; U[(i0+0)*m+(j0+1)][col] = 1
        V[(j0+1)*p+(k0+1)][col] = 1
        W[(k0+0)*n+(i0+0)][col] = -1; W[(k0+1)*n+(i0+0)][col] = 1
        col += 1
        # M6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
        U[(i0+1)*m+(j0+0)][col] = 1; U[(i0+0)*m+(j0+0)][col] = -1
        V[(j0+0)*p+(k0+0)][col] = 1; V[(j0+0)*p+(k0+1)][col] = 1
        W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        # M7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
        U[(i0+0)*m+(j0+1)][col] = 1; U[(i0+1)*m+(j0+1)][col] = -1
        V[(j0+1)*p+(k0+0)][col] = 1; V[(j0+1)*p+(k0+1)][col] = 1
        W[(k0+0)*n+(i0+0)][col] = 1
        col += 1
        return col
    
    # Partition: n=[1,2], m=[2,1,2], p=[2,1,2]
    # n_offsets: [0, 1], m_offsets: [0, 2, 3], p_offsets: [0, 2, 3]
    n_parts = [1, 2]
    m_parts = [2, 1, 2]
    p_parts = [2, 1, 2]
    n_off = [0, 1]
    m_off = [0, 2, 3]
    p_off = [0, 2, 3]
    
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
                    col = strassen2(i0, j0, k0, U, V, W, col)
                else:
                    col = schoolbook(i0, j0, k0, ns, ms, ps, U, V, W, col)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    if time.time() > _DEADLINE:
        return None
    return {"U": U, "V": V, "W": W}
