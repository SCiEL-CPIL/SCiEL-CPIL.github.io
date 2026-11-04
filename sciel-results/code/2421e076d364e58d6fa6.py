import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition is valid."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check shapes
    if U_arr.shape != (n * m, rank) or V_arr.shape != (m * p, rank) or W_arr.shape != (p * n, rank):
        return False
        
    # Check all (i,j,k) entries equal 1
    # Use einsum for efficiency: T_ijk = sum_r U_im_jr V_jp_kr W_kn_ir
    # We want T_ijk == 1 for all i,j,k
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute tensor
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    if np.allclose(T, 1.0, atol=tol):
        return True
    return False

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _local_search(n, m, p, rank, seed, deadline):
    """
    Local search using block coordinate descent with analytic updates.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize U, V, W with small random values
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # Target tensor T is all ones
    T = np.ones((n, m, p))
    T_flat = T.flatten()
    
    # Precompute indices for mapping
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Function to update U given V, W
    def update_U(V, W):
        # We want to solve T_flat = M_U @ U_flat
        # M_U has columns corresponding to U_flat entries
        # M_U[t, (i*m+j, r)] = V[j*p+k, r] * W[k*n+i, r] where t = i*m*p + j*p + k
        # This is a large matrix. Let's construct it.
        # Size: (n*m*p) x (n*m*rank)
        M_U = np.zeros((n * m * p, n * m * rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t_idx = i * m * p + j * p + k
                    u_row = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(rank):
                        u_col = u_row * rank + r
                        M_U[t_idx, u_col] = V[v_idx, r] * W[w_idx, r]
        
        # Solve least squares
        U_flat, _, _, _ = np.linalg.lstsq(M_U, T_flat, rcond=None)
        return U_flat.reshape(n * m, rank)

    # Function to update V given U, W
    def update_V(U, W):
        M_V = np.zeros((n * m * p, m * p * rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t_idx = i * m * p + j * p + k
                    v_row = j * p + k
                    u_idx = i * m + j
                    w_idx = k * n + i
                    for r in range(rank):
                        v_col = v_row * rank + r
                        M_V[t_idx, v_col] = U[u_idx, r] * W[w_idx, r]
        
        V_flat, _, _, _ = np.linalg.lstsq(M_V, T_flat, rcond=None)
        return V_flat.reshape(m * p, rank)

    # Function to update W given U, V
    def update_W(U, V):
        M_W = np.zeros((n * m * p, p * n * rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t_idx = i * m * p + j * p + k
                    w_row = k * n + i
                    u_idx = i * m + j
                    v_idx = j * p + k
                    for r in range(rank):
                        w_col = w_row * rank + r
                        M_W[t_idx, w_col] = U[u_idx, r] * V[v_idx, r]
        
        W_flat, _, _, _ = np.linalg.lstsq(M_W, T_flat, rcond=None)
        return W_flat.reshape(p * n, rank)

    # Iterate
    for _ in range(50): # Max iterations
        if time.time() > deadline - 10:
            break
        
        U = update_U(V, W)
        V = update_V(U, W)
        W = update_W(U, V)
        
        # Check convergence
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        err = np.linalg.norm(T_calc - T)
        
        if err < 1e-6:
            break
    
    # Round to half-integers
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    # If rounding didn't work, try to refine by solving for exact integers/half-integers?
    # Or just return None
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook verification failed")
    
    # Try to find a better rank
    # We try ranks from best_rank - 1 down to a lower bound
    # Lower bound: max(n*m, m*p, p*n) = 24
    # But we know Strassen-like algorithms can do better than schoolbook.
    # Let's try a range of ranks.
    
    # To save time, we don't try every rank. We try a few promising ones.
    # If we find a rank r, we return it.
    
    # Strategy: Try ranks 71, 70, 69, ... down to 60.
    # If we find one, return.
    
    seeds = [0, 1, 2, 3, 4]
    
    for r in range(best_rank - 1, 50, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in seeds:
            if time.time() > deadline - 10:
                break
            
            result = _local_search(n, m, p, r, seed, deadline)
            if result is not None:
                U_new, V_new, W_new = result
                if _verify(U_new, V_new, W_new, n, m, p):
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    best_rank = r
                    found = True
                    break
        
        if found:
            # Found a better rank, can we go lower?
            # Yes, continue the loop to try r-1
            pass
    
    return best_result
