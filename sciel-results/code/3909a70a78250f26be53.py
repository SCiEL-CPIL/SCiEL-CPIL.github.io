"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i*m+j, j*p+k, k*n+i]
    
    # For each (i,j,k), check if the sum equals 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = 0.0
                for c in range(rank):
                    s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _random_search(start_rank, end_rank, time_budget):
    """Search for a decomposition with rank in [start_rank, end_rank]."""
    n, m, p = DIMENSIONS
    best = None
    
    for r in range(start_rank, end_rank + 1):
        if time.time() > time_budget:
            break
        # Try several random initializations
        for attempt in range(50):
            if time.time() > time_budget:
                break
            
            # Random integer/half-integer initialization
            U = np.random.randint(-2, 3, size=(n*m, r)).astype(float) / 2.0
            V = np.random.randint(-2, 3, size=(m*p, r)).astype(float) / 2.0
            W = np.random.randint(-2, 3, size=(p*n, r)).astype(float) / 2.0
            
            # Verify
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                if best is None or r < best[0]:
                    best = (r, U.copy(), V.copy(), W.copy())
                break  # Found one for this rank, move to next rank
    
    return best

def _greedy_reduce(base_U, base_V, base_W):
    """Try to reduce rank by removing columns that are linearly dependent in the tensor sense."""
    n, m, p = DIMENSIONS
    U = np.array(base_U, dtype=float)
    V = np.array(base_V, dtype=float)
    W = np.array(base_W, dtype=float)
    rank = U.shape[1]
    
    # Try to remove each column and check if the decomposition still holds
    # This is expensive, so we do it carefully
    for col_idx in range(rank - 1, -1, -1):
        if rank <= 1:
            break
        # Try removing this column
        U_test = np.delete(U, col_idx, axis=1)
        V_test = np.delete(V, col_idx, axis=1)
        W_test = np.delete(W, col_idx, axis=1)
        
        # Quick check: does the decomposition still hold?
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(rank - 1):
                        s += U_test[u_idx, c] * V_test[v_idx, c] * W_test[w_idx, c]
                    if abs(s - 1.0) > 1e-6:
                        valid = False
                        break
        
        if valid:
            U = U_test
            V = V_test
            W = W_test
            rank -= 1
    
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    sb = _schoolbook()
    best_U, best_V, best_W = sb["U"], sb["V"], sb["W"]
    best_rank = schoolbook_rank
    
    # Try greedy reduction on schoolbook
    U_r, V_r, W_r, r_r = _greedy_reduce(best_U, best_V, best_W)
    if r_r < best_rank:
        best_U, best_V, best_W = U_r.tolist(), V_r.tolist(), W_r.tolist()
        best_rank = r_r
    
    # Try random search for lower ranks
    if time.time() < deadline:
        result = _random_search(max(20, best_rank - 10), best_rank, deadline)
        if result is not None:
            r, U_r, V_r, W_r = result
            if r < best_rank:
                best_U, best_V, best_W = U_r.tolist(), V_r.tolist(), W_r.tolist()
                best_rank = r
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
