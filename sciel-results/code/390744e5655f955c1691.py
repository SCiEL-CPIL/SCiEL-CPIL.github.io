import numpy as np
import scipy.optimize as opt
import time

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k
    """
    rank = U.shape[1]
    # Convert to numpy arrays for speed
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                # Compute the sum
                val = np.dot(U_np[idx_u, :], V_np[idx_v, :] * W_np[idx_w, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
    return True

def _find_rank11_2x3x3():
    """
    Attempts to find a rank 11 decomposition for 2x3x3 multiplication.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = 2, 3, 3
    rank = 11
    
    # Define the residual function
    def residuals(params):
        # params is a flat array of size rank * (n*m + m*p + p*n)
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U_flat = params[0:size_u*rank]
        V_flat = params[size_u*rank : size_u*rank + size_v*rank]
        W_flat = params[size_u*rank + size_v*rank : size_u*rank + size_v*rank + size_w*rank]
        
        U = U_flat.reshape((size_u, rank))
        V = V_flat.reshape((size_v, rank))
        W = W_flat.reshape((size_w, rank))
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    # Term: U[idx_u, :] . (V[idx_v, :] * W[idx_w, :])
                    val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                    res.append(val - 1.0)
        return np.array(res)

    # Initial guess
    # Try to start with small random values
    np.random.seed(42)
    
    start_time = time.time()
    timeout = 10.0 # seconds
    
    best_result = None
    best_err = float('inf')
    
    for attempt in range(5):
        if time.time() - start_time > timeout:
            break
            
        # Random initialization
        # Scale: entries are likely small
        init_size = (n*m + m*p + p*n) * rank
        x0 = np.random.normal(0, 1.0, init_size)
        
        try:
            res = opt.least_squares(residuals, x0, method='lm', max_nfev=2000)
            err = res.cost
            if err < best_err:
                best_err = err
                best_result = res.x
                
            if err < 1e-10:
                break
        except Exception:
            continue
            
    if best_result is None or best_err > 1e-8:
        return None
        
    # Extract U, V, W
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    U_flat = best_result[0:size_u*rank]
    V_flat = best_result[size_u*rank : size_u*rank + size_v*rank]
    W_flat = best_result[size_u*rank + size_v*rank : size_u*rank + size_v*rank + size_w*rank]
    
    U = U_flat.reshape((size_u, rank))
    V = V_flat.reshape((size_v, rank))
    W = W_flat.reshape((size_w, rank))
    
    # Round to nearest half-integer
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Verify rounded solution
    if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
        return U_rounded, V_rounded, W_rounded
    else:
        # Try rounding to integer?
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if _verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
            return U_int, V_int, W_int
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try to find Rank 22 decomposition using two Rank 11 blocks
    sub_decomp = _find_rank11_2x3x3()
    
    if sub_decomp is not None:
        U_sub, V_sub, W_sub = sub_decomp
        rank_sub = U_sub.shape[1]
        total_rank = 2 * rank_sub
        
        # Construct full U, V, W
        # U: (n*m, total_rank)
        U_full = np.zeros((n * m, total_rank))
        # V: (m*p, total_rank)
        V_full = np.zeros((m * p, total_rank))
        # W: (p*n, total_rank)
        W_full = np.zeros((p * n, total_rank))
        
        # Block 1 (k=0..2)
        U_full[:, 0:rank_sub] = U_sub
        
        # Map V_sub (3x3 indices j, k) to V_full (3x6 indices j, k)
        for j in range(m):
            for k in range(3): # k is 0,1,2
                idx_v_sub = j * 3 + k
                idx_v_full = j * 6 + k
                V_full[idx_v_full, 0:rank_sub] = V_sub[idx_v_sub, :]
                
        # Map W_sub (3x2 indices k, i) to W_full (6x2 indices k, i)
        for k in range(3):
            for i in range(2):
                idx_w_sub = k * 2 + i
                idx_w_full = k * 2 + i
                W_full[idx_w_full, 0:rank_sub] = W_sub[idx_w_sub, :]
                
        # Block 2 (k=3..5)
        U_full[:, rank_sub:2*rank_sub] = U_sub
        
        # Map V_sub to V_full for k=3,4,5
        for j in range(m):
            for k in range(3): # k is 0,1,2 in sub, maps to 3,4,5 in full
                idx_v_sub = j * 3 + k
                idx_v_full = j * 6 + (k + 3)
                V_full[idx_v_full, rank_sub:2*rank_sub] = V_sub[idx_v_sub, :]
                
        # Map W_sub to W_full for k=3,4,5
        for k in range(3):
            for i in range(2):
                idx_w_sub = k * 2 + i
                idx_w_full = (k + 3) * 2 + i
                W_full[idx_w_full, rank_sub:2*rank_sub] = W_sub[idx_w_sub, :]
                
        # Verify full decomposition
        if _verify_decomposition(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to Naive Rank 36
    U = [[0] * 36 for _ in range(n * m)]
    V = [[0] * 36 for _ in range(m * p)]
    W = [[0] * 36 for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
