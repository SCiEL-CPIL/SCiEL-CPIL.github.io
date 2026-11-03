import time
import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [2, 3, 8]

def _build_initial_decomposition(n, m, p):
    """Build the rank-44 initial decomposition using Strassen + schoolbook."""
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Part 1: 2x2 * 2x8 using Strassen on 2-column blocks
    for j in range(4):
        w_r00 = 4 * j
        w_r01 = 4 * j + 2
        w_r10 = 4 * j + 1
        w_r11 = 4 * j + 3
        v_e = 2 * j
        v_f = 2 * j + 1
        v_g = 8 + 2 * j
        v_h = 9 + 2 * j
        
        # P1 = (a+d)(e+h)
        U[0, col] = 1; U[4, col] = 1
        V[v_e, col] = 1; V[v_h, col] = 1
        W[w_r00, col] = 1; W[w_r11, col] = 1
        col += 1
        # P2 = (c+d)e
        U[3, col] = 1; U[4, col] = 1
        V[v_e, col] = 1
        W[w_r10, col] = 1; W[w_r11, col] = -1
        col += 1
        # P3 = a(f-h)
        U[0, col] = 1
        V[v_f, col] = 1; V[v_h, col] = -1
        W[w_r01, col] = 1; W[w_r11, col] = 1
        col += 1
        # P4 = d(g-e)
        U[4, col] = 1
        V[v_g, col] = 1; V[v_e, col] = -1
        W[w_r00, col] = 1; W[w_r10, col] = 1
        col += 1
        # P5 = (a+b)h
        U[0, col] = 1; U[1, col] = 1
        V[v_h, col] = 1
        W[w_r00, col] = -1; W[w_r01, col] = 1
        col += 1
        # P6 = (c-a)(e+f)
        U[3, col] = 1; U[0, col] = -1
        V[v_e, col] = 1; V[v_f, col] = 1
        W[w_r11, col] = 1
        col += 1
        # P7 = (b-d)(g+h)
        U[1, col] = 1; U[4, col] = -1
        V[v_g, col] = 1; V[v_h, col] = 1
        W[w_r00, col] = 1
        col += 1
    
    # Part 2: 2x1 * 1x8 schoolbook
    for i in range(2):
        for k in range(8):
            u_idx = i * 3 + 2
            v_idx = 2 * 8 + k
            w_idx = k * 2 + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
    
    return U, V, W

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-10):
    """Vectorized verification using NumPy."""
    # Reshape for einsum: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # We need to check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    
    # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)
    
    # Compute T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
    
    return np.allclose(T, 1.0, atol=tol)

def _try_reduce_rank(U, V, W, n, m, p, max_iterations=200, time_budget=100):
    """
    Attempt to reduce the rank by finding linear dependencies among the rank-1 terms.
    Uses vectorized verification for speed.
    """
    start_time = time.time()
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = current_U.shape[1]
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_budget:
            break
        if current_rank <= 40:
            break
            
        successful_elimination = False
        
        # Try removing each column
        for col_to_remove in range(current_rank - 1, -1, -1):
            if time.time() - start_time > time_budget:
                break
                
            # Create a copy without this column
            mask = np.ones(current_rank, dtype=bool)
            mask[col_to_remove] = False
            new_U = current_U[:, mask]
            new_V = current_V[:, mask]
            new_W = current_W[:, mask]
            
            # Verify if the remaining decomposition still works
            if _verify_decomposition_fast(new_U, new_V, new_W, n, m, p, tol=1e-8):
                current_U, current_V, current_W = new_U, new_V, new_W
                current_rank -= 1
                successful_elimination = True
                break
        
        if not successful_elimination:
            break
            
    return current_U, current_V, current_W, current_rank

def construct():
    """
    Constructs a low-rank matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Build initial rank-44 decomposition
    U, V, W = _build_initial_decomposition(n, m, p)
    current_rank = U.shape[1]
    
    # Verify initial decomposition
    if not _verify_decomposition_fast(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = np.zeros((n * m, rank))
        V_fb = np.zeros((m * p, rank))
        W_fb = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U_fb[i * m + j, idx] = 1
                    V_fb[j * p + k, idx] = 1
                    W_fb[k * n + i, idx] = 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to reduce rank with remaining time
    time_remaining = deadline - time.time() - 5
    if time_remaining > 5:
        U, V, W, new_rank = _try_reduce_rank(U, V, W, n, m, p, time_budget=time_remaining)
        if new_rank < current_rank:
            current_rank = new_rank
            if _verify_decomposition_fast(U, V, W, n, m, p):
                current_rank = new_rank
            else:
                U, V, W = _build_initial_decomposition(n, m, p)
                current_rank = 44
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
