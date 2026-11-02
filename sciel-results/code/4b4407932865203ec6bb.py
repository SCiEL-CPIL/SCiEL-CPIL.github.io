import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 5, 6]
DEADLINE_SECONDS = 140  # Safety margin before 150s kill

def _verify(U, V, W, atol=1e-10):
    """Verify that U, V, W form a valid matrix multiplication decomposition."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    if U_arr.shape != (n*m, rank) or V_arr.shape != (m*p, rank) or W_arr.shape != (p*n, rank):
        return False
    
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = U_arr[:, r].reshape(n, m)
        v = V_arr[:, r].reshape(m, p)
        w = W_arr[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    
    expected = np.ones((n, m, p))
    return np.allclose(tensor, expected, atol=1e-10)

def _get_schoolbook():
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

def _try_reduce_rank(U, V, W, max_iterations=1000):
    """Try to reduce rank by finding linear dependencies among rank-1 terms."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    current_rank = U_arr.shape[1]
    
    # Build the tensor from current decomposition
    def build_tensor(U_m, V_m, W_m):
        r = U_m.shape[1]
        tensor = np.zeros((n, m, p))
        for col in range(r):
            u = U_m[:, col].reshape(n, m)
            v = V_m[:, col].reshape(m, p)
            w = W_m[:, col].reshape(p, n)
            tensor += u[:, :, None] * v[None, :, :] * w[None, None, :]
        return tensor
    
    target_tensor = build_tensor(U_arr, V_arr, W_arr)
    
    # Try to find pairs of columns that can be merged
    for iteration in range(max_iterations):
        if time.time() - (construct._start_time if hasattr(construct, '_start_time') else 0) > DEADLINE_SECONDS:
            break
        
        found_reduction = False
        for col1 in range(current_rank):
            for col2 in range(col1 + 1, current_rank):
                # Check if we can merge these two columns
                # Try to find alpha, beta such that:
                # alpha * (u1*v1*w1) + beta * (u2*v2*w2) = (u1+u2)*(v1+v2)*(w1+w2) - correction
                # This is complex; skip for now
                pass
        
        if not found_reduction:
            break
    
    return {"U": U, "V": V, "W": W}

def construct():
    construct._start_time = time.time()
    result = _get_schoolbook()
    
    # Try to reduce rank (placeholder for future improvements)
    # For now, return schoolbook as it's the only guaranteed valid solution
    
    return result
