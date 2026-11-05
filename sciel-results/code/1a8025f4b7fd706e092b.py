import time
import numpy as np

DIMENSIONS = [3, 5, 5]
_DEADLINE = time.time() + 140

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Target rank estimate
    # We will use schoolbook as a safe base and try to apply Strassen where possible.
    
    def schoolbook_full():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    try:
        # Attempt the optimized block decomposition
        rank = 71
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        
        def schoolbook_block(i0, j0, k0, n_s, m_s, p_s):
            nonlocal col
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

        def strassen2_block(i0, j0, k0):
            nonlocal col
            # M1 = (A00+A11)(B00+B11)
            U[i0*m + j0, col] = 1; U[(i0+1)*m + (j0+1), col] = 1
            V[j0*p + k0, col] = 1; V[(j0+1)*p + (k0+1), col] = 1
            W[k0*n + i0, col] = 1; W[(k0+1)*n + (i0+1), col] = 1
            col += 1
            
            # M2 = (A10+A11)(B00)
            U[(i0+1)*m + j0, col] = 1; U[(i0+1)*m + (j0+1), col] = 1
            V[j0*p + k0, col] = 1
            W[k0*n + (i0+1), col] = 1; W[(k0+1)*n + (i0+1), col] = -1
            col += 1
            
            # M3 = (A00)(B01-B11)
            U[i0*m + j0, col] = 1
            V[j0*p + (k0+1), col] = 1; V[(j0+1)*p + (k0+1), col] = -1
            W[(k0+1)*n + i0, col] = 1; W[(k0+1)*n + (i0+1), col] = 1
            col += 1
            
            # M4 = (A11)(B10-B00)
            U[(i0+1)*m + (j0+1), col] = 1
            V[(j0+1)*p + k0, col] = 1; V[j0*p + k0, col] = -1
            W[k0*n + i0, col] = 1; W[k0*n + (i0+1), col] = 1
            col += 1
            
            # M5 = (A00+A01)(B11)
            U[i0*m + j0, col] = 1; U[i0*m + (j0+1), col] = 1
            V[(j0+1)*p + (k0+1), col] = 1
            W[k0*n + i0, col] = -1; W[(k0+1)*n + i0, col] = 1
            col += 1
            
            # M6 = (A10-A00)(B00+B01)
            U[(i0+1)*m + j0, col] = 1; U[i0*m + j0, col] = -1
            V[j0*p + k0, col] = 1; V[j0*p + (k0+1), col] = 1
            W[(k0+1)*n + (i0+1), col] = 1
            col += 1
            
            # M7 = (A01-A11)(B10+B11)
            U[i0*m + (j0+1), col] = 1; U[(i0+1)*m + (j0+1), col] = -1
            V[(j0+1)*p + k0, col] = 1; V[(j0+1)*p + (k0+1), col] = 1
            W[k0*n + i0, col] = 1
            col += 1

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
                        strassen2_block(i0, j0, k0)
                    else:
                        schoolbook_block(i0, j0, k0, ns, ms, ps)
        
        if col != rank:
            raise ValueError("Internal rank mismatch")

        if time.time() > _DEADLINE:
            return None
        
        return {"U": U, "V": V, "W": W}

    except Exception:
        # Fallback to schoolbook if anything goes wrong
        U, V, W, rank = schoolbook_full()
        if time.time() > _DEADLINE:
            return None
        return {"U": U, "V": V, "W": W}
