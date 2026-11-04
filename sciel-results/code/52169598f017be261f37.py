"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _fast_verify(n, m, p, U, V, W, tol=1e-8):
    """Fast vectorized verification of decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Build index arrays
    i_idx = np.repeat(np.arange(n), m * p).reshape(-1)
    j_idx = np.tile(np.repeat(np.arange(m), p), n).reshape(-1)
    k_idx = np.tile(np.arange(p), n * m).reshape(-1)
    
    # For each (i,j,k), compute sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Vectorize by computing all at once
    errors = np.zeros(n * m * p)
    for c in range(rank):
        u_c = U_arr[:, c]
        v_c = V_arr[:, c]
        w_c = W_arr[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    errors[idx] += u_c[i * m + j] * v_c[j * p + k] * w_c[k * n + i]
    
    max_err = np.max(np.abs(errors - 1.0))
    return max_err < tol, max_err

def _try_strassen_block(n, m, p, deadline):
    """
    Try to decompose 3x4x5 using block Strassen-like approach.
    Split into 2x2x2 blocks (rank 7 each) + scalar blocks (rank 1 each).
    
    3 = 2+1, 4 = 2+2, 5 = 2+2+1
    Blocks:
    - (0:2, 0:2, 0:2): 2x2x2 Strassen, rank 7
    - (0:2, 0:2, 2:4): 2x2x2 Strassen, rank 7  
    - (0:2, 2:4, 0:2): 2x2x2 Strassen, rank 7
    - (0:2, 2:4, 2:4): 2x2x2 Strassen, rank 7
    - (2:3, 0:2, 0:2): 1x2x2, rank 2
    - (2:3, 0:2, 2:4): 1x2x2, rank 2
    - (2:3, 2:4, 0:2): 1x2x2, rank 2
    - (2:3, 2:4, 2:4): 1x2x2, rank 2
    - (0:2, 0:2, 4:5): 2x2x1, rank 2
    - (0:2, 2:4, 4:5): 2x2x1, rank 2
    - (2:3, 0:2, 4:5): 1x2x1, rank 1
    - (2:3, 2:4, 4:5): 1x2x1, rank 1
    
    Total: 4*7 + 4*2 + 4*2 + 2*1 = 28 + 8 + 8 + 2 = 46
    """
    if time.time() > deadline:
        return None
    
    n, m, p = 3, 4, 5
    U_rows = n * m  # 12
    V_rows = m * p  # 20
    W_rows = p * n  # 15
    
    # Strassen 2x2x2 decomposition
    # For 2x2 matrices A, B: C = A*B
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # We'll build the decomposition block by block
    # Each block contributes to specific (i,j,k) entries
    
    total_rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(i_start, i_end, j_start, j_end, k_start, k_end, block_type):
        nonlocal total_rank, U_cols, V_cols, W_cols
        
        ni = i_end - i_start
        nj = j_end - j_start
        nk = k_end - k_start
        
        if block_type == 'strassen' and ni == 2 and nj == 2 and nk == 2:
            # 2x2x2 Strassen: rank 7
            for s in range(7):
                u_col = [0.0] * U_rows
                v_col = [0.0] * V_rows
                w_col = [0.0] * W_rows
                
                # Define the 7 products
                if s == 0:  # M1 = (A00+A11)(B00+B11)
                    # U: A00 and A11 get +1
                    u_col[i_start * m + j_start] = 1.0
                    u_col[(i_start+1) * m + (j_start+1)] = 1.0
                    # V: B00 and B11 get +1
                    v_col[j_start * p + k_start] = 1.0
                    v_col[(j_start+1) * p + (k_start+1)] = 1.0
                    # W: C00 gets +1, C11 gets +1
                    w_col[k_start * n + i_start] = 1.0
                    w_col[(k_start+1) * n + (i_start+1)] = 1.0
                elif s == 1:  # M2 = (A10+A11)B00
                    u_col[(i_start+1) * m + j_start] = 1.0
                    u_col[(i_start+1) * m + (j_start+1)] = 1.0
                    v_col[j_start * p + k_start] = 1.0
                    w_col[k_start * n + (i_start+1)] = 1.0
                elif s == 2:  # M3 = A00(B10-B11)
                    u_col[i_start * m + j_start] = 1.0
                    v_col[(j_start+1) * p + k_start] = 1.0
                    v_col[(j_start+1) * p + (k_start+1)] = -1.0
                    w_col[k_start * n + i_start] = 1.0
                elif s == 3:  # M4 = A11(B01-B10)
                    u_col[(i_start+1) * m + (j_start+1)] = 1.0
                    v_col[j_start * p + (k_start+1)] = 1.0
                    v_col[(j_start+1) * p + k_start] = -1.0
                    w_col[k_start * n + i_start] = 1.0
                    w_col[(k_start+1) * n + (i_start+1)] = 1.0
                elif s == 4:  # M5 = (A00+A01)B11
                    u_col[i_start * m + j_start] = 1.0
                    u_col[i_start * m + (j_start+1)] = 1.0
                    v_col[(j_start+1) * p + (k_start+1)] = 1.0
                    w_col[(k_start+1) * n + i_start] = 1.0
                    w_col[k_start * n + i_start] = -1.0
                elif s == 5:  # M6 = (A10-A00)(B00+B10)
                    u_col[(i_start+1) * m + j_start] = 1.0
                    u_col[i_start * m + j_start] = -1.0
                    v_col[j_start * p + k_start] = 1.0
                    v_col[(j_start+1) * p + k_start] = 1.0
                    w_col[k_start * n + (i_start+1)] = 1.0
                elif s == 6:  # M7 = (A01-A11)(B01+B11)
                    u_col[i_start * m + (j_start+1)] = 1.0
                    u_col[(i_start+1) * m + (j_start+1)] = -1.0
                    v_col[j_start * p + (k_start+1)] = 1.0
                    v_col[(j_start+1) * p + (k_start+1)] = 1.0
                    w_col[k_start * n + i_start] = -1.0
                    w_col[(k_start+1) * n + i_start] = 1.0
                    w_col[(k_start+1) * n + (i_start+1)] = 1.0
                
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)
                total_rank += 1
        
        elif ni == 1 and nj == 2 and nk == 2:
            # 1x2x2: rank 2 (just two scalar multiplications)
            for j_off in range(2):
                for k_off in range(2):
                    u_col = [0.0] * U_rows
                    v_col = [0.0] * V_rows
                    w_col = [0.0] * W_rows
                    u_col[i_start * m + j_start + j_off] = 1.0
                    v_col[(j_start + j_off) * p + k_start + k_off] = 1.0
                    w_col[(k_start + k_off) * n + i_start] = 1.0
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)
                    total_rank += 1
        
        elif ni == 2 and nj == 2 and nk == 1:
            # 2x2x1: rank 2
            for i_off in range(2):
                for j_off in range(2):
                    u_col = [0.0] * U_rows
                    v_col = [0.0] * V_rows
                    w_col = [0.0] * W_rows
                    u_col[(i_start + i_off) * m + j_start + j_off] = 1.0
                    v_col[(j_start + j_off) * p + k_start] = 1.0
                    w_col[k_start * n + i_start + i_off] = 1.0
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)
                    total_rank += 1
        
        elif ni == 1 and nj == 2 and nk == 1:
            # 1x2x1: rank 1
            for j_off in range(2):
                u_col = [0.0] * U_rows
                v_col = [0.0] * V_rows
                w_col = [0.0] * W_rows
                u_col[i_start * m + j_start + j_off] = 1.0
                v_col[(j_start + j_off) * p + k_start] = 1.0
                w_col[k_start * n + i_start] = 1.0
                U_cols.append(u_col)
                V_cols.append(v_col)
                W_cols.append(w_col)
                total_rank += 1
    
    # Define the blocks
    # 2x2x2 Strassen blocks
    add_block(0, 2, 0, 2, 0, 2, 'strassen')
    add_block(0, 2, 0, 2, 2, 4, 'strassen')
    add_block(0, 2, 2, 4, 0, 2, 'strassen')
    add_block(0, 2, 2, 4, 2, 4, 'strassen')
    
    # 1x2x2 blocks
    add_block(2, 3, 0, 2, 0, 2, 'scalar')
    add_block(2, 3, 0, 2, 2, 4, 'scalar')
    add_block(2, 3, 2, 4, 0, 2, 'scalar')
    add_block(2, 3, 2, 4, 2, 4, 'scalar')
    
    # 2x2x1 blocks
    add_block(0, 2, 0, 2, 4, 5, 'scalar')
    add_block(0, 2, 2, 4, 4, 5, 'scalar')
    
    # 1x2x1 blocks
    add_block(2, 3, 0, 2, 4, 5, 'scalar')
    add_block(2, 3, 2, 4, 4, 5, 'scalar')
    
    if time.time() > deadline:
        return None
    
    # Build matrices
    U_mat = np.array(U_cols).T  # shape (U_rows, total_rank)
    V_mat = np.array(V_cols).T  # shape (V_rows, total_rank)
    W_mat = np.array(W_cols).T  # shape (W_rows, total_rank)
    
    # Verify
    valid, max_err = _fast_verify(n, m, p, U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), tol=1e-8)
    if valid:
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    
    return None

def _schoolbook(n, m, p):
    """Standard schoolbook multiplication, rank = n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Try structured Strassen-block decomposition
    result = _try_strassen_block(n, m, p, deadline)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
