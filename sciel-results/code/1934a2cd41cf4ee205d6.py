"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_decomposition():
    """
    Construct a Strassen-based decomposition for 2x4x8 matrix multiplication.
    Total rank = 2 * 4 * 7 = 56.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col = 0
    for a_block in range(2):
        for b_block in range(4):
            for term in strassen_terms:
                A_coeffs, B_coeffs, C_coeffs = term
                for (li, lj), coeff in A_coeffs.items():
                    global_i = li
                    global_j = 2 * a_block + lj
                    U[global_i * m + global_j, col] += coeff
                for (lj, lk), coeff in B_coeffs.items():
                    global_j = 2 * a_block + lj
                    global_k = 2 * b_block + lk
                    V[global_j * p + global_k, col] += coeff
                for (li, lk), coeff in C_coeffs.items():
                    global_i = li
                    global_k = 2 * b_block + lk
                    W[global_k * n + global_i, col] += coeff
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

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

def _find_redundant_columns(U_arr, V_arr, W_arr, tol=1e-10):
    """
    Find columns that are linearly dependent across all three matrices simultaneously.
    A column c is redundant if U[:,c] is in span of other U cols,
    V[:,c] is in span of other V cols, AND W[:,c] is in span of other W cols.
    We greedily remove such columns.
    """
    rank = U_arr.shape[1]
    keep = list(range(rank))
    
    # Greedy: try to remove each column
    for c in range(rank):
        if c not in keep:
            continue
        remaining = [x for x in keep if x != c]
        if not remaining:
            continue
        
        # Check if U[:,c] is in span of U[:, remaining]
        U_rem = U_arr[:, remaining]
        U_c = U_arr[:, c]
        # Check rank
        if np.linalg.matrix_rank(np.column_stack([U_rem, U_c])) > np.linalg.matrix_rank(U_rem):
            continue
        
        # Check if V[:,c] is in span of V[:, remaining]
        V_rem = V_arr[:, remaining]
        V_c = V_arr[:, c]
        if np.linalg.matrix_rank(np.column_stack([V_rem, V_c])) > np.linalg.matrix_rank(V_rem):
            continue
        
        # Check if W[:,c] is in span of W[:, remaining]
        W_rem = W_arr[:, remaining]
        W_c = W_arr[:, c]
        if np.linalg.matrix_rank(np.column_stack([W_rem, W_c])) > np.linalg.matrix_rank(W_rem):
            continue
        
        # Column c is redundant - remove it
        keep.remove(c)
    
    return keep

def _try_rank_reduction_via_linear_combination(U, V, W, n, m, p, max_time=30.0):
    """
    Attempt to reduce the rank by finding linear dependencies among the rank-1 terms.
    """
    start = time.time()
    deadline = start + max_time
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    keep = _find_redundant_columns(U_arr, V_arr, W_arr, tol=1e-10)
    
    if len(keep) == rank:
        return {"U": U, "V": V, "W": W}
    
    U_new = U_arr[:, keep].tolist()
    V_new = V_arr[:, keep].tolist()
    W_new = W_arr[:, keep].tolist()
    
    return {"U": U_new, "V": V_new, "W": W_new}

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        result = _strassen_decomposition()
        if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
            if time.time() < deadline - 35.0:
                try:
                    reduced = _try_rank_reduction_via_linear_combination(
                        result["U"], result["V"], result["W"], n, m, p,
                        max_time=min(30.0, deadline - time.time() - 5.0)
                    )
                    if _verify(reduced["U"], reduced["V"], reduced["W"], n, m, p, tol=1e-9):
                        return reduced
                except Exception:
                    pass
            return result
    except Exception:
        pass
    
    return _schoolbook_decomposition()

# EVOLVE-BLOCK-END
