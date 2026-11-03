import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Target Rank 41 decomposition
    # Split: n=1+2, m=2+2, p=2+5
    # Blocks:
    # 11: 1x2x2 (Rank 4)
    # 12: 1x2x5 (Rank 10)
    # 21: 2x2x2 (Rank 7, Strassen)
    # 22: 2x2x5 (Rank 20)
    
    n1, n2 = 1, 2
    m1, m2 = 2, 2
    p1, p2 = 2, 5
    
    # Initialize matrices
    total_rank = 4 + 10 + 7 + 20
    U = np.zeros((n*m, total_rank), dtype=np.float64)
    V = np.zeros((m*p, total_rank), dtype=np.float64)
    W = np.zeros((p*n, total_rank), dtype=np.float64)
    
    col = 0
    
    # Helper to add schoolbook block
    def add_schoolbook(n_s, m_s, p_s, offset_i, offset_j, offset_k, col_start):
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    U[(offset_i + i) * m + (offset_j + j), col_start] = 1
                    V[(offset_j + j) * p + (offset_k + k), col_start] = 1
                    W[(offset_k + k) * n + (offset_i + i), col_start] = 1
                    col_start += 1
        return col_start

    # Helper to add Strassen block (2x2x2)
    def add_strassen(offset_i, offset_j, offset_k, col_start):
        # n_s=2, m_s=2, p_s=2
        # Local indices i', j', k' in 0..1
        # Global i = offset_i + i', j = offset_j + j', k = offset_k + k'
        
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        U[(offset_i+0)*m + (offset_j+0), col_start] = 1
        U[(offset_i+1)*m + (offset_j+1), col_start] = 1
        V[(offset_j+0)*p + (offset_k+0), col_start] = 1
        V[(offset_j+1)*p + (offset_k+1), col_start] = 1
        W[(offset_k+0)*n + (offset_i+0), col_start] = 1
        W[(offset_k+1)*n + (offset_i+1), col_start] = 1
        col_start += 1
        
        # M2 = (A10+A11)B00 -> C10
        U[(offset_i+1)*m + (offset_j+0), col_start] = 1
        U[(offset_i+1)*m + (offset_j+1), col_start] = 1
        V[(offset_j+0)*p + (offset_k+0), col_start] = 1
        W[(offset_k+0)*n + (offset_i+1), col_start] = 1
        col_start += 1
        
        # M3 = A00(B00-B10) -> C01
        U[(offset_i+0)*m + (offset_j+0), col_start] = 1
        V[(offset_j+0)*p + (offset_k+0), col_start] = 1
        V[(offset_j+1)*p + (offset_k+0), col_start] = -1
        W[(offset_k+0)*n + (offset_i+0), col_start] = 1
        col_start += 1
        
        # M4 = A11(B10-B00) -> C00, C10
        U[(offset_i+1)*m + (offset_j+1), col_start] = 1
        V[(offset_j+1)*p + (offset_k+0), col_start] = 1
        V[(offset_j+0)*p + (offset_k+0), col_start] = -1
        W[(offset_k+0)*n + (offset_i+0), col_start] = 1
        W[(offset_k+0)*n + (offset_i+1), col_start] = 1
        col_start += 1
        
        # M5 = (A00+A01)B11 -> C00(-), C01(+)
        U[(offset_i+0)*m + (offset_j+0), col_start] = 1
        U[(offset_i+0)*m + (offset_j+1), col_start] = 1
        V[(offset_j+1)*p + (offset_k+1), col_start] = 1
        W[(offset_k+0)*n + (offset_i+0), col_start] = -1
        W[(offset_k+1)*n + (offset_i+0), col_start] = 1
        col_start += 1
        
        # M6 = (A10-A00)(B00+B01) -> C11
        U[(offset_i+1)*m + (offset_j+0), col_start] = 1
        U[(offset_i+0)*m + (offset_j+0), col_start] = -1
        V[(offset_j+0)*p + (offset_k+0), col_start] = 1
        V[(offset_j+0)*p + (offset_k+1), col_start] = 1
        W[(offset_k+1)*n + (offset_i+1), col_start] = 1
        col_start += 1
        
        # M7 = (A01-A11)(B10+B11) -> C00
        U[(offset_i+0)*m + (offset_j+1), col_start] = 1
        U[(offset_i+1)*m + (offset_j+1), col_start] = -1
        V[(offset_j+1)*p + (offset_k+0), col_start] = 1
        V[(offset_j+1)*p + (offset_k+1), col_start] = 1
        W[(offset_k+0)*n + (offset_i+0), col_start] = 1
        col_start += 1
        
        return col_start

    # Block 11: 1x2x2, offset i=0, j=0, k=0
    col = add_schoolbook(n1, m1, p1, 0, 0, 0, col)
    
    # Block 12: 1x2x5, offset i=0, j=2, k=2
    col = add_schoolbook(n1, m2, p2, 0, m1, p1, col)
    
    # Block 21: 2x2x2, offset i=1, j=0, k=0
    col = add_strassen(n1, m1, p1, col)
    
    # Block 22: 2x2x5, offset i=1, j=2, k=2
    col = add_schoolbook(n2, m2, p2, n1, m1, p1, col)
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break

    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
