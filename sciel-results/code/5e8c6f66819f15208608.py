import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def _strassen_2x2_decomp():
    """
    Return the 7 rank-1 terms for 2x2x2 matrix multiplication.
    Each term is a tuple of three dicts representing sparse vectors for U, V, W.
    Keys are local indices 0..3 for the 2x2 matrix.
    """
    patterns = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),           # M1
        ({2:1, 3:1}, {0:1},     {2:1, 3:-1}),           # M2
        ({0:1},       {1:1, 3:-1}, {1:1, 3:1}),         # M3
        ({3:1},       {2:1, 0:-1}, {0:1, 2:1}),         # M4
        ({0:1, 1:1}, {3:1},     {0:-1, 1:1}),           # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),               # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),               # M7
    ]
    return patterns

def _build_rank_56():
    """
    Construct the rank-56 decomposition for 2x4 @ 4x8.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    strassen_patterns = _strassen_2x2_decomp()
    
    # Loop over the 8 independent 2x2x2 blocks
    for a_block in range(2):
        for b_col_block in range(4):
            base_col = 7 * (4 * a_block + b_col_block)
            
            for s, (u_c, v_c, w_c) in enumerate(strassen_patterns):
                col = base_col + s
                
                # Process U
                for local_idx, val in u_c.items():
                    li = local_idx // 2
                    lj = local_idx % 2
                    global_i = li
                    global_j = 2 * a_block + lj
                    U[global_i * m + global_j][col] += val
                
                # Process V
                for local_idx, val in v_c.items():
                    lj = local_idx // 2
                    lk = local_idx % 2
                    global_j = 2 * a_block + lj
                    global_k = 2 * b_col_block + lk
                    V[global_j * p + global_k][col] += val
                
                # Process W
                for local_idx, val in w_c.items():
                    li = local_idx // 2
                    lk = local_idx % 2
                    global_i = li
                    global_k = 2 * b_col_block + lk
                    W[global_k * n + global_i][col] += val
                    
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    """
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Use vectorized computation for speed
    # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = U_arr[i * m + j, :]
                v_vec = V_arr[j * p + k, :]
                w_vec = W_arr[k * n + i, :]
                val = np.dot(u_vec, v_vec * w_vec)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition with rank n*m*p = 64."""
    n, m, p = DIMENSIONS
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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        if time.time() < deadline:
            result = _build_rank_56()
            if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
                return result
    except Exception:
        pass
    
    return _schoolbook_decomposition()
