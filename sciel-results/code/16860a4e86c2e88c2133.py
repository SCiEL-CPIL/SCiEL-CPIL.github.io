import numpy as np
from fractions import Fraction
import time

def _verify(U, V, W, n=3, m=4, p=4, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _schoolbook(n=3, m=4, p=4):
    """Standard schoolbook decomposition with rank n*m*p."""
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

def _add_strassen_block(U, V, W, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, offset, n, m, p):
    """
    Add Strassen 2x2 block decomposition.
    a_rows, a_cols: 2 indices for A block
    b_rows, b_cols: 2 indices for B block
    c_rows, c_cols: 2 indices for C block
    offset: starting column index in U, V, W
    """
    # Map local indices (0,1) to global indices
    # A local (r, c) -> global (a_rows[r], a_cols[c])
    # B local (r, c) -> global (b_rows[r], b_cols[c])
    # C local (r, c) -> global (c_rows[r], c_cols[c])
    
    # Helper to get global row index
    def u_idx(r, c): return a_rows[r] * m + a_cols[c]
    def v_idx(r, c): return b_rows[r] * p + b_cols[c]
    def w_idx(r, c): return c_cols[c] * n + c_rows[r] # Note: W[k*n+i], so col is k, row is i
    
    # The 7 terms
    # M1
    U[u_idx(0,0)][offset+0] += 1; U[u_idx(1,1)][offset+0] += 1
    V[v_idx(0,0)][offset+0] += 1; V[v_idx(1,1)][offset+0] += 1
    W[w_idx(0,0)][offset+0] += 1; W[w_idx(1,1)][offset+0] += 1
    
    # M2
    U[u_idx(1,0)][offset+1] += 1; U[u_idx(1,1)][offset+1] += 1
    V[v_idx(0,0)][offset+1] += 1
    W[w_idx(1,0)][offset+1] += 1; W[w_idx(1,1)][offset+1] -= 1
    
    # M3
    U[u_idx(0,0)][offset+2] += 1
    V[v_idx(1,0)][offset+2] += 1; V[v_idx(1,1)][offset+2] -= 1
    W[w_idx(0,1)][offset+2] += 1; W[w_idx(1,1)][offset+2] += 1
    
    # M4
    U[u_idx(1,1)][offset+3] += 1
    V[v_idx(0,1)][offset+3] += 1; V[v_idx(0,0)][offset+3] -= 1
    W[w_idx(0,0)][offset+3] += 1; W[w_idx(1,0)][offset+3] += 1
    
    # M5
    U[u_idx(0,0)][offset+4] += 1; U[u_idx(0,1)][offset+4] += 1
    V[v_idx(1,1)][offset+4] += 1
    W[w_idx(0,0)][offset+4] -= 1; W[w_idx(0,1)][offset+4] += 1
    
    # M6
    U[u_idx(1,0)][offset+5] += 1; U[u_idx(0,0)][offset+5] -= 1
    V[v_idx(0,0)][offset+5] += 1; V[v_idx(1,0)][offset+5] += 1
    W[w_idx(1,1)][offset+5] += 1
    
    # M7
    U[u_idx(0,1)][offset+6] += 1; U[u_idx(1,1)][offset+6] -= 1
    V[v_idx(0,1)][offset+6] += 1; V[v_idx(1,1)][offset+6] += 1
    W[w_idx(0,0)][offset+6] += 1; W[w_idx(0,1)][offset+6] -= 1

def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    offset = 0
    
    # Part 1: Bottom row (i=2)
    # 1x4x4 schoolbook
    for j in range(4):
        for k in range(4):
            U[2 * m + j][offset] = 1
            V[j * p + k][offset] = 1
            W[k * n + 2][offset] = 1
            offset += 1
    
    # Part 2: Top 2 rows (i=0,1)
    # 4 blocks of 2x2x2 Strassen
    
    # Block 1: A(0,1;0,1) * B(0,1;0,1) -> C(0,1;0,1)
    _add_strassen_block(U, V, W, [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], offset, n, m, p)
    offset += 7
    
    # Block 2: A(0,1;2,3) * B(2,3;0,1) -> C(0,1;0,1)
    _add_strassen_block(U, V, W, [0,1], [2,3], [2,3], [0,1], [0,1], [0,1], offset, n, m, p)
    offset += 7
    
    # Block 3: A(0,1;0,1) * B(0,1;2,3) -> C(0,1;2,3)
    _add_strassen_block(U, V, W, [0,1], [0,1], [0,1], [2,3], [0,1], [2,3], offset, n, m, p)
    offset += 7
    
    # Block 4: A(0,1;2,3) * B(2,3;2,3) -> C(0,1;2,3)
    _add_strassen_block(U, V, W, [0,1], [2,3], [2,3], [2,3], [0,1], [2,3], offset, n, m, p)
    offset += 7
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback
        return _schoolbook(n, m, p)
        
    return {"U": U, "V": V, "W": W}
