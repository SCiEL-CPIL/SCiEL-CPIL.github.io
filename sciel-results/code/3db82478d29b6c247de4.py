import numpy as np
from itertools import product
import time

def construct():
    n, m, p = 2, 3, 3
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Try ranks from 10 upward
    for rank in range(10, 19):
        if time.time() > deadline - 5:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback to schoolbook (rank 18)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank decomposition with integer/half-integer entries."""
    np.random.seed(42)
    
    for attempt in range(500):
        if time.time() > deadline - 3:
            return None
        
        # Initialize with sparse random patterns in {-1, 0, 1}
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for c in range(rank):
            # Sparse initialization
            nnz_U = np.random.randint(1, 4)
            nnz_V = np.random.randint(1, 4)
            nnz_W = np.random.randint(1, 4)
            
            u_idx = np.random.choice(n * m, nnz_U, replace=False)
            v_idx = np.random.choice(m * p, nnz_V, replace=False)
            w_idx = np.random.choice(p * n, nnz_W, replace=False)
            
            U[u_idx, c] = np.random.choice([-1.0, 1.0], nnz_U)
            V[v_idx, c] = np.random.choice([-1.0, 1.0], nnz_V)
            W[w_idx, c] = np.random.choice([-1.0, 1.0], nnz_W)
        
        if _check_decomposition(n, m, p, U, V, W):
            return _to_lists(U, V, W)
        
        # Local search: try flipping individual entries
        for _ in range(50):
            if time.time() > deadline - 3:
                return None
            improved = False
            
            # Try flipping U entries
            for c in range(rank):
                for row in range(n * m):
                    old = U[row, c]
                    new_vals = [v for v in [-1.0, 0.0, 1.0] if v != old]
                    for new_val in new_vals:
                        U[row, c] = new_val
                        if _check_decomposition(n, m, p, U, V, W):
                            return _to_lists(U, V, W)
                        improved = True
                    U[row, c] = old
            
            # Try flipping V entries
            for c in range(rank):
                for row in range(m * p):
                    old = V[row, c]
                    new_vals = [v for v in [-1.0, 0.0, 1.0] if v != old]
                    for new_val in new_vals:
                        V[row, c] = new_val
                        if _check_decomposition(n, m, p, U, V, W):
                            return _to_lists(U, V, W)
                        improved = True
                    V[row, c] = old
            
            # Try flipping W entries
            for c in range(rank):
                for row in range(p * n):
                    old = W[row, c]
                    new_vals = [v for v in [-1.0, 0.0, 1.0] if v != old]
                    for new_val in new_vals:
                        W[row, c] = new_val
                        if _check_decomposition(n, m, p, U, V, W):
                            return _to_lists(U, V, W)
                        improved = True
                    W[row, c] = old
            
            if not improved:
                break
    
    return None

def _check_decomposition(n, m, p, U, V, W):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for l in range(p):
                    # Sum over c: U[i*m+j, c] * V[jp*p+l, c] * W[l*n+i, c]
                    s = np.sum(U[i * m + j, :] * V[jp * p + l, :] * W[l * n + i, :])
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _to_lists(U, V, W):
    """Convert numpy arrays to lists of lists."""
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
