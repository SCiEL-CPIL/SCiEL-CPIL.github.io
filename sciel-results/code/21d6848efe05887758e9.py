import time
import numpy as np

DIMENSIONS = [3, 5, 5]
_DEADLINE = time.time() + 140

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # We use a block decomposition strategy.
    # Partition n=[1,2], m=[2,1,2], p=[2,1,2]
    # This yields 4 blocks of size 2x2x2 where we can use Strassen (rank 7).
    # The rest are schoolbook.
    # Total rank: 71.
    
    rank = 71
    
    U = np.zeros((n * m, rank), dtype=float)  # 15 x 71
    V = np.zeros((m * p, rank), dtype=float)  # 25 x 71
    W = np.zeros((p * n, rank), dtype=float)  # 15 x 71
    
    col = 0
    
    def schoolbook(i0, j0, k0, n_s, m_s, p_s, U, V, W, col):
        for i_s in range(n_s):
            for j_s in range(m_s):
                for k_s in range(p_s):
                    i = i0 + i_s
                    j = j0 + j_s
                    k = k0 + k_s
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return col
    
    def strassen2(i0, j0, k0, U, V, W, col):
        # Strassen's algorithm for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        U[(i0+0)*m+(j0+0), col] = 1; U[(i0+1)*m+(j0+1), col] = 1
        V[(j0+0)*p+(k0+0), col] = 1; V[(j0+1)*p+(k0+1), col] = 1
        W[(k0+0)*n+(i0+0), col] = 1; W[(k0+1)*n+(i0+1), col] = 1
        col += 1
        
        # M2 = (A10+A11)(B00)
        U[(i0+1)*m+(j0+0), col] = 1; U[(i0+1)*m+(j0+1), col] = 1
        V[(j0+0)*p+(k0+0), col] = 1
        W[(k0+0)*n+(i0+1), col] = 1; W[(k0+1)*n+(i0+1), col] = -1
        col += 1
        
        # M3 = (A00)(B01-B11)
        U[(i0+0)*m+(j0+0), col] = 1
        V[(j0+0)*p+(k0+1), col] = 1; V[(j0+1)*p+(k0+1), col] = -1
        W[(k0+1)*n+(i0+0), col] = 1; W[(k0+1)*n+(i0+1), col] = 1
        col += 1
        
        # M4 = (A11)(B10-B00)
        U[(i0+1)*m+(j0+1), col] = 1
        V[(j0+1)*p+(k0+0), col] = 1; V[(j0+0)*p+(k0+0), col] = -1
        W[(k0+0)*n+(i0+0), col] = 1; W[(k0+0)*n+(i0+1), col] = 1
        col += 1
        
        # M5 = (A00+A01)(B11)
        U[(i0+0)*m+(j0+0), col] = 1; U[(i0+0)*m+(j0+1), col] = 1
        V[(j0+1)*p+(k0+1), col] = 1
        W[(k0+0)*n+(i0+0), col] = -1; W[(k0+1)*n+(i0+0), col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i0+1)*m+(j0+0), col] = 1; U[(i0+0)*m+(j0+0), col] = -1
        V[(j0+0)*p+(k0+0), col] = 1; V[(j0+0)*p+(k0+1), col] = 1
        W[(k0+1)*n+(i0+1), col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[(i0+0)*m+(j0+1), col] = 1; U[(i0+1)*m+(j0+1), col] = -1
        V[(j0+1)*p+(k0+0), col] = 1; V[(j0+1)*p+(k0+1), col] = 1
        W[(k0+0)*n+(i0+0), col] = 1
        col += 1
        
        return col

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
    
    if col != rank:
        # Fallback to schoolbook if something went wrong (should not happen)
        U = np.zeros((n * m, n * m * p), dtype=float)
        V = np.zeros((m * p, n * m * p), dtype=float)
        W = np.zeros((p * n, n * m * p), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        rank = n * m * p

    if time.time() > _DEADLINE:
        return None
    return {"U": U, "V": V, "W": W}
