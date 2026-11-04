"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    """Block decomposition with Strassen on 2x2x2 sub-blocks for 3x5x7."""
    start_time = time.time()
    deadline = start_time + 100  # safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Split dimensions: rows 2+1, middle 2+2+1, cols 2+2+2+1
    row_splits = [2, 1]
    mid_splits = [2, 2, 1]
    col_splits = [2, 2, 2, 1]
    
    row_starts = [0]
    for s in row_splits:
        row_starts.append(row_starts[-1] + s)
    mid_starts = [0]
    for s in mid_splits:
        mid_starts.append(mid_starts[-1] + s)
    col_starts = [0]
    for s in col_splits:
        col_starts.append(col_starts[-1] + s)
    
    # Collect all rank-1 terms: (U_row, V_row, W_row, coeff)
    terms = []
    
    def add_strassen_2x2x2(r_start, m_start, p_start):
        """Add 7 Strassen terms for a 2x2x2 block."""
        i0, i1 = r_start, r_start + 1
        j0, j1 = m_start, m_start + 1
        k0, k1 = p_start, p_start + 1
        
        # The 7 Strassen rank-1 terms for 2x2x2
        # Each term: (U_coeffs, V_coeffs, W_coeffs)
        # U indexed by (i,j), V by (j,k), W by (k,i)
        
        # M1 = (A00+A11)(B00+B11)
        # U: (i0,j0)=1, (i1,j1)=1
        # V: (j0,k0)=1, (j1,k1)=1
        # W: (k0,i0)=1, (k1,i1)=1
        u1 = {(i0,j0): 1, (i1,j1): 1}
        v1 = {(j0,k0): 1, (j1,k1): 1}
        w1 = {(k0,i0): 1, (k1,i1): 1}
        
        # M2 = (A10+A11)B00
        # U: (i1,j0)=1, (i1,j1)=1
        # V: (j0,k0)=1
        # W: (k0,i1)=1, (k1,i1)=-1
        u2 = {(i1,j0): 1, (i1,j1): 1}
        v2 = {(j0,k0): 1}
        w2 = {(k0,i1): 1, (k1,i1): -1}
        
        # M3 = A00(B10-B11)
        # U: (i0,j0)=1
        # V: (j1,k0)=1, (j1,k1)=-1
        # W: (k0,i0)=1, (k1,i1)=1
        u3 = {(i0,j0): 1}
        v3 = {(j1,k0): 1, (j1,k1): -1}
        w3 = {(k0,i0): 1, (k1,i1): 1}
        
        # M4 = A11(B01-B00)
        # U: (i1,j1)=1
        # V: (j0,k1)=1, (j0,k0)=-1
        # W: (k0,i0)=1, (k0,i1)=1
        u4 = {(i1,j1): 1}
        v4 = {(j0,k1): 1, (j0,k0): -1}
        w4 = {(k0,i0): 1, (k0,i1): 1}
        
        # M5 = (A00+A01)B11
        # U: (i0,j0)=1, (i0,j1)=1
        # V: (j1,k1)=1
        # W: (k0,i0)=-1, (k0,i1)=1
        u5 = {(i0,j0): 1, (i0,j1): 1}
        v5 = {(j1,k1): 1}
        w5 = {(k0,i0): -1, (k0,i1): 1}
        
        # M6 = (A10-A00)(B00+B01)
        # U: (i1,j0)=1, (i0,j0)=-1
        # V: (j0,k0)=1, (j0,k1)=1
        # W: (k1,i1)=1
        u6 = {(i1,j0): 1, (i0,j0): -1}
        v6 = {(j0,k0): 1, (j0,k1): 1}
        w6 = {(k1,i1): 1}
        
        # M7 = (A01-A11)(B10+B11)
        # U: (i0,j1)=1, (i1,j1)=-1
        # V: (j1,k0)=1, (j1,k1)=1
        # W: (k0,i0)=1
        u7 = {(i0,j1): 1, (i1,j1): -1}
        v7 = {(j1,k0): 1, (j1,k1): 1}
        w7 = {(k0,i0): 1}
        
        for uc, vc, wc in [(u1,v1,w1), (u2,v2,w2), (u3,v3,w3), 
                           (u4,v4,w4), (u5,v5,w5), (u6,v6,w6), (u7,v7,w7)]:
            for (ii, jj), uval in uc.items():
                for (jj2, kk), vval in vc.items():
                    if jj == jj2:
                        for (kk2, ii2), wval in wc.items():
                            if kk == kk2:
                                terms.append((ii * m + jj, jj * p + kk, kk * n + ii2, uval * vval * wval))
    
    for ri, (r_start, r_size) in enumerate(zip(row_starts[:-1], row_splits)):
        for mi, (m_start, m_size) in enumerate(zip(mid_starts[:-1], mid_splits)):
            for pi, (p_start, p_size) in enumerate(zip(col_starts[:-1], col_splits)):
                if r_size == 2 and m_size == 2 and p_size == 2:
                    add_strassen_2x2x2(r_start, m_start, p_start)
                else:
                    # Schoolbook for this block
                    for i in range(r_start, r_start + r_size):
                        for j in range(m_start, m_start + m_size):
                            for k in range(p_start, p_start + p_size):
                                terms.append((i * m + j, j * p + k, k * n + i, 1))
    
    # Combine terms: group by (U_row, V_row, W_row) and sum coefficients
    combined = defaultdict(int)
    for (u_row, v_row, w_row, coeff) in terms:
        combined[(u_row, v_row, w_row)] += coeff
    
    # Remove zero coefficients
    combined = {k: v for k, v in combined.items() if v != 0}
    
    rank = len(combined)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, ((u_row, v_row, w_row), coeff) in enumerate(combined.items()):
        U[u_row][col] = coeff
        V[v_row][col] = coeff
        W[w_row][col] = coeff
    
    # Verify
    if time.time() < deadline:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-9:
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
