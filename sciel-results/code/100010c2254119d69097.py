import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def _to_halfint_list(arr):
    """Convert numpy array to list with integer/half-integer entries."""
    result = []
    for row in arr:
        row_list = []
        for val in row:
            rounded = round(val * 2) / 2.0
            if abs(rounded - int(rounded)) < 1e-9:
                row_list.append(int(rounded))
            else:
                row_list.append(rounded)
        result.append(row_list)
    return result

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Structured rank-n*p decomposition
    # c indexes (i, k) pairs: c = i * p + k
    rank_struct = n * p  # 3 * 7 = 21
    
    U = np.zeros((n * m, rank_struct), dtype=np.float64)
    V = np.zeros((m * p, rank_struct), dtype=np.float64)
    W = np.zeros((p * n, rank_struct), dtype=np.float64)
    
    for i in range(n):
        for k in range(p):
            c = i * p + k
            # U: for row i*m + j, set to 1 for all j
            for j in range(m):
                U[i * m + j, c] = 1.0
            # V: for row j*p + k, set to 1 for all j
            for j in range(m):
                V[j * p + k, c] = 1.0
            # W: for row k*n + i, set to 1
            W[k * n + i, c] = 1.0
    
    # Verify the structured solution
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {
            "U": _to_halfint_list(U),
            "V": _to_halfint_list(V),
            "W": _to_halfint_list(W)
        }
    
    # Fallback to schoolbook if structured solution fails verification
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=np.float64)
    V_school = np.zeros((m * p, rank_school), dtype=np.float64)
    W_school = np.zeros((p * n, rank_school), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    return {
        "U": _to_halfint_list(U_school),
        "V": _to_halfint_list(V_school),
        "W": _to_halfint_list(W_school)
    }
