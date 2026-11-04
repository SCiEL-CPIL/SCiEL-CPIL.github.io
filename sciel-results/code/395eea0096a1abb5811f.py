import time
import numpy as np

DIMENSIONS = [4, 5, 5]
_start_time = time.time()
TIME_LIMIT = 140  # safety margin

def _check_time():
    if time.time() - _start_time > TIME_LIMIT:
        raise TimeoutError()

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2 * 2x2, rank 7."""
    U = np.array([
        [1, 0, 0, 1],   # a+d
        [0, 0, 1, 1],   # c+d
        [1, 0, 0, 0],   # a
        [0, 0, 0, 1],   # d
        [1, 1, 0, 0],   # a+b
        [-1, 0, 1, 0],  # c-a
        [0, 1, 0, -1]   # b-d
    ])
    V = np.array([
        [1, 0, 0, 1],   # e+h
        [1, 0, 0, 0],   # e
        [0, 1, 0, -1],  # f-h
        [-1, 0, 1, 0],  # g-e
        [0, 0, 0, 1],   # h
        [1, 1, 0, 0],   # e+f
        [0, 0, 1, 1]    # g+h
    ])
    # W[k*m+i, r]
    W = np.array([
        [1, 0, 0, 1, -1, 0, -1],   # C11
        [0, 0, 1, 0, 1, 0, 0],     # C12
        [0, 1, 0, 1, 0, 0, 0],     # C21
        [1, -1, 1, 0, 0, 1, 0]     # C22
    ])
    return (U, V, W)

def _schoolbook(m, n, p):
    """Return schoolbook decomposition for m x n * n x p."""
    rank = m * n * p
    U = np.zeros((m * n, rank))
    V = np.zeros((n * p, rank))
    W = np.zeros((p * m, rank))
    col = 0
    for i in range(m):
        for j in range(n):
            for k in range(p):
                U[i * n + j, col] = 1
                V[j * p + k, col] = 1
                W[k * m + i, col] = 1
                col += 1
    return (U, V, W)

def _combine(U1, V1, W1, m1, n1, p1, offset_m, offset_n, offset_p, m, n, p):
    """Add a sub-decomposition to the full-size tensors using NumPy."""
    rank = U1.shape[1]
    U_new = np.zeros((m * n, rank))
    V_new = np.zeros((n * p, rank))
    W_new = np.zeros((p * m, rank))
    
    # Extract sub-blocks
    # U1 is (m1*n1, rank)
    # We need to place it in U_new at rows (offset_m..offset_m+m1-1) x (offset_n..offset_n+n1-1)
    # This is a block diagonal placement.
    
    for r in range(rank):
        # U
        u_block = U1[:, r]
        # Reshape to (m1, n1)
        u_mat = u_block.reshape((m1, n1))
        # Place in U_new
        U_new[offset_m* n + offset_n : (offset_m+m1)*n + offset_n : n, :] # This slice is wrong for non-contiguous blocks
        # Actually, U_new is flat (m*n, rank).
        # Indices: (offset_m + i) * n + (offset_n + j)
        
        # It is easier to iterate or use advanced indexing
        rows = (offset_m + np.arange(m1))[:, None] * n + (offset_n + np.arange(n1))[None, :]
        rows = rows.flatten()
        U_new[rows, r] = u_block
        
        # V
        v_block = V1[:, r]
        v_mat = v_block.reshape((n1, p1))
        rows_v = (offset_n + np.arange(n1))[:, None] * p + (offset_p + np.arange(p1))[None, :]
        rows_v = rows_v.flatten()
        V_new[rows_v, r] = v_block
        
        # W
        w_block = W1[:, r]
        w_mat = w_block.reshape((p1, m1))
        rows_w = (offset_p + np.arange(p1))[:, None] * m + (offset_m + np.arange(m1))[None, :]
        rows_w = rows_w.flatten()
        W_new[rows_w, r] = w_block
        
    return (U_new, V_new, W_new)

def _decompose(m, n, p):
    """
    Return (U, V, W) for m×n * n×p matrix multiplication.
    U: shape (m*n, rank)
    V: shape (n*p, rank)
    W: shape (p*m, rank), W[k*m+i, r] convention
    """
    _check_time()
    
    # Base case: empty
    if m == 0 or n == 0 or p == 0:
        return (np.array([]), np.array([]), np.array([]))
    
    # Degenerate cases
    if m == 1 and n == 1 and p == 1:
        return (np.array([[1]]), np.array([[1]]), np.array([[1]]))
        
    if m == 1:
        rank = n * p
        U = np.zeros((n, rank))
        V = np.zeros((n * p, rank))
        W = np.zeros((p, rank))
        col = 0
        for j in range(n):
            for k in range(p):
                U[j, col] = 1
                V[j * p + k, col] = 1
                W[k, col] = 1
                col += 1
        return (U, V, W)
        
    if p == 1:
        rank = m * n
        U = np.zeros((m * n, rank))
        V = np.zeros((n, rank))
        W = np.zeros((m, rank))
        col = 0
        for i in range(m):
            for j in range(n):
                U[i * n + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        return (U, V, W)
        
    if n == 1:
        rank = m * p
        U = np.zeros((m, rank))
        V = np.zeros((p, rank))
        W = np.zeros((m * p, rank))
        col = 0
        for i in range(m):
            for k in range(p):
                U[i, col] = 1
                V[k, col] = 1
                W[k * m + i, col] = 1
                col += 1
        return (U, V, W)

    # Special case: Strassen for 2×2 * 2×2
    if m == 2 and n == 2 and p == 2:
        return _strassen_2x2x2()
    
    # Try all possible splits and pick the best
    best = None
    best_rank = m * n * p  # schoolbook
    
    for m1 in range(1, m):
        m2 = m - m1
        _check_time()
        for n1 in range(1, n):
            n2 = n - n1
            _check_time()
            for p1 in range(1, p):
                p2 = p - p1
                _check_time()
                try:
                    # 8 sub-decompositions
                    U1, V1, W1 = _decompose(m1, n1, p1)
                    U2, V2, W2 = _decompose(m1, n1, p2)
                    U3, V3, W3 = _decompose(m1, n2, p1)
                    U4, V4, W4 = _decompose(m1, n2, p2)
                    U5, V5, W5 = _decompose(m2, n1, p1)
                    U6, V6, W6 = _decompose(m2, n1, p2)
                    U7, V7, W7 = _decompose(m2, n2, p1)
                    U8, V8, W8 = _decompose(m2, n2, p2)
                    
                    total_rank = (U1.shape[1] + U2.shape[1] + U3.shape[1] + U4.shape[1] +
                                 U5.shape[1] + U6.shape[1] + U7.shape[1] + U8.shape[1])
                    if total_rank < best_rank:
                        best_rank = total_rank
                        U_new = np.zeros((m * n, total_rank))
                        V_new = np.zeros((n * p, total_rank))
                        W_new = np.zeros((p * m, total_rank))
                        
                        col_offset = 0
                        
                        # A11*B11: m1×n1 * n1×p1
                        u_sub, v_sub, w_sub = _combine(U1, V1, W1, m1, n1, p1, 0, 0, 0, m, n, p)
                        U_new[:, col_offset:col_offset+U1.shape[1]] = u_sub
                        V_new[:, col_offset:col_offset+U1.shape[1]] = v_sub
                        W_new[:, col_offset:col_offset+U1.shape[1]] = w_sub
                        col_offset += U1.shape[1]
                        
                        # A12*B21: m1×n2 * n2×p1
                        u_sub, v_sub, w_sub = _combine(U3, V3, W3, m1, n2, p1, 0, n1, 0, m, n, p)
                        U_new[:, col_offset:col_offset+U3.shape[1]] = u_sub
                        V_new[:, col_offset:col_offset+U3.shape[1]] = v_sub
                        W_new[:, col_offset:col_offset+U3.shape[1]] = w_sub
                        col_offset += U3.shape[1]
                        
                        # A11*B12: m1×n1 * n1×p2
                        u_sub, v_sub, w_sub = _combine(U2, V2, W2, m1, n1, p2, 0, 0, p1, m, n, p)
                        U_new[:, col_offset:col_offset+U2.shape[1]] = u_sub
                        V_new[:, col_offset:col_offset+U2.shape[1]] = v_sub
                        W_new[:, col_offset:col_offset+U2.shape[1]] = w_sub
                        col_offset += U2.shape[1]
                        
                        # A12*B22: m1×n2 * n2×p2
                        u_sub, v_sub, w_sub = _combine(U4, V4, W4, m1, n2, p2, 0, n1, p1, m, n, p)
                        U_new[:, col_offset:col_offset+U4.shape[1]] = u_sub
                        V_new[:, col_offset:col_offset+U4.shape[1]] = v_sub
                        W_new[:, col_offset:col_offset+U4.shape[1]] = w_sub
                        col_offset += U4.shape[1]
                        
                        # A21*B11: m2×n1 * n1×p1
                        u_sub, v_sub, w_sub = _combine(U5, V5, W5, m2, n1, p1, m1, 0, 0, m, n, p)
                        U_new[:, col_offset:col_offset+U5.shape[1]] = u_sub
                        V_new[:, col_offset:col_offset+U5.shape[1]] = v_sub
                        W_new[:, col_offset:col_offset+U5.shape[1]] = w_sub
                        col_offset += U5.shape[1]
                        
                        # A21*B12: m2×n1 * n1×p2
                        u_sub, v_sub, w_sub = _combine(U6, V6, W6, m2, n1, p2, m1, 0, p1, m, n, p)
                        U_new[:, col_offset:col_offset+U6.shape[1]] = u_sub
                        V_new[:, col_offset:col_offset+U6.shape[1]] = v_sub
                        W_new[:, col_offset:col_offset+U6.shape[1]] = w_sub
                        col_offset += U6.shape[1]
                        
                        # A22*B21: m2×n2 * n2×p1
                        u_sub, v_sub, w_sub = _combine(U7, V7, W7, m2, n2, p1, m1, n1, 0, m, n, p)
                        U_new[:, col_offset:col_offset+U7.shape[1]] = u_sub
                        V_new[:, col_offset:col_offset+U7.shape[1]] = v_sub
                        W_new[:, col_offset:col_offset+U7.shape[1]] = w_sub
                        col_offset += U7.shape[1]
                        
                        # A22*B22: m2×n2 * n2×p2
                        u_sub, v_sub, w_sub = _combine(U8, V8, W8, m2, n2, p2, m1, n1, p1, m, n, p)
                        U_new[:, col_offset:col_offset+U8.shape[1]] = u_sub
                        V_new[:, col_offset:col_offset+U8.shape[1]] = v_sub
                        W_new[:, col_offset:col_offset+U8.shape[1]] = w_sub
                        col_offset += U8.shape[1]
                        
                        best = (U_new, V_new, W_new)
                except TimeoutError:
                    raise
                except Exception:
                    continue
    
    if best is not None:
        return best
    else:
        return _schoolbook(m, n, p)

def construct():
    global _start_time
    _start_time = time.time()
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    U, V, W = _decompose(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
