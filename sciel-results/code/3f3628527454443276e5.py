import numpy as np
import time

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def _elapsed():
    return time.time() - START_TIME

def _deadline_reached():
    return _elapsed() > TIME_LIMIT

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[3, 4] = -1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x4x6_decomposition():
    """
    Build the 2x4x6 matrix multiplication decomposition using Strassen on 2x2x2 blocks.
    Rank = 42.
    """
    n, m, p = 2, 4, 6
    n_j_blocks = m // 2  # 2
    n_k_blocks = p // 2  # 3
    rank = n_j_blocks * n_k_blocks * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    for jb in range(n_j_blocks):
        for kb in range(n_k_blocks):
            base_col = (jb * n_k_blocks + kb) * 7
            j0 = 2 * jb
            j1 = 2 * jb + 1
            k0 = 2 * kb
            k1 = 2 * kb + 1
            
            for c in range(7):
                col = base_col + c
                for i in range(2):
                    for j_local in range(2):
                        j = j0 + j_local
                        U[i * m + j, col] = U_s[i * 2 + j_local, c]
                for j_local in range(2):
                    j = j0 + j_local
                    for k_local in range(2):
                        k = k0 + k_local
                        V[j * p + k, col] = V_s[j_local * 2 + k_local, c]
                for k_local in range(2):
                    k = k0 + k_local
                    for i in range(2):
                        W[k * n + i, col] = W_s[k_local * 2 + i, c]
    
    return U, V, W

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify the decomposition is exact."""
    n, m, p = 2, 4, 6
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    
    recon = np.einsum('ac,bc,dc->abd', U_arr, V_arr, W_arr)
    
    error = np.linalg.norm(target - recon)
    return error < tol, error

def _try_rank_41_als():
    """
    Attempt to find a rank 41 decomposition via ALS with multiple seeds.
    This is a structural change from the previous random search:
    1. Uses proper ALS updates (solving linear systems per factor)
    2. Multiple random restarts with different seeds
    3. Early termination if error stops improving
    """
    n, m, p = 2, 4, 6
    rank = 41
    
    # Precompute target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    
    target_flat = target.reshape(-1)
    
    # Precompute index mappings for ALS
    # For U update: for each a (i*m+j), we need to solve U[a,:] such that
    # sum_c U[a,c] * V[b,c] * W[d,c] = target[a,b,d] for all b,d
    # This is: U[a,:] @ P = target[a,:,:].reshape(-1) where P[c, b, d] = V[b,c] * W[d,c]
    
    # For V update: for each b (j*p+k), V[b,:] @ Q = target[:,b,:].reshape(-1)
    # where Q[c, a, d] = U[a,c] * W[d,c]
    
    # For W update: for each d (k*n+i), W[d,:] @ R = target[:,:,d].reshape(-1)
    # where R[c, a, b] = U[a,c] * V[b,c]
    
    best_error = float('inf')
    best_result = None
    
    for seed in range(10):
        if _deadline_reached():
            break
        
        rng = np.random.RandomState(seed)
        
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        prev_error = float('inf')
        no_improve_count = 0
        
        for iteration in range(50):
            if _deadline_reached():
                break
            
            # Update U
            M = V[:, None, :] * W[None, :, :]  # shape (m*p, p*n, rank)
            P_flat = M.reshape(rank, -1)  # shape (rank, m*p*p*n)
            P_T = P_flat.T  # shape (m*p*p*n, rank)
            
            # Solve for U: U[a,:] = target[a,:,:].reshape(-1) @ pinv(P_T)
            try:
                P_pinv = np.linalg.pinv(P_T)
                new_U = np.zeros((n * m, rank))
                for a in range(n * m):
                    y = target[a, :, :].reshape(-1)
                    new_U[a, :] = y @ P_pinv
                U = new_U
            except:
                break
            
            if _deadline_reached():
                break
            
            # Update V
            Q = U[:, None, :] * W[None, :, :]  # shape (n*m, p*n, rank)
            Q_flat = Q.reshape(rank, -1)  # shape (rank, n*m*p*n)
            Q_T = Q_flat.T  # shape (n*m*p*n, rank)
            
            try:
                Q_pinv = np.linalg.pinv(Q_T)
                new_V = np.zeros((m * p, rank))
                for b in range(m * p):
                    y = target[:, b, :].reshape(-1)
                    new_V[b, :] = y @ Q_pinv
                V = new_V
            except:
                break
            
            if _deadline_reached():
                break
            
            # Update W
            R = U[:, None, :] * V[None, :, :]  # shape (n*m, m*p, rank)
            R_flat = R.reshape(rank, -1)  # shape (rank, n*m*m*p)
            R_T = R_flat.T  # shape (n*m*m*p, rank)
            
            try:
                R_pinv = np.linalg.pinv(R_T)
                new_W = np.zeros((p * n, rank))
                for d in range(p * n):
                    y = target[:, :, d].reshape(-1)
                    new_W[d, :] = y @ R_pinv
                W = new_W
            except:
                break
            
            # Compute error
            recon = np.einsum('ac,bc,dc->abd', U, V, W)
            error = np.linalg.norm(target - recon)
            
            if error < best_error:
                best_error = error
                best_result = (U.copy(), V.copy(), W.copy())
            
            if error < 1e-9:
                return U, V, W
            
            if error >= prev_error * 0.999:
                no_improve_count += 1
            else:
                no_improve_count = 0
            
            prev_error = error
            
            if no_improve_count > 5:
                break
        
        if best_error < 1e-9 and best_result is not None:
            return best_result
    
    return best_result if best_error < 1e-9 else None

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Construct the guaranteed Rank 42 solution
    U42, V42, W42 = _build_2x4x6_decomposition()
    
    # Convert to lists for return
    U42_list = U42.tolist()
    V42_list = V42.tolist()
    W42_list = W42.tolist()
    
    # 2. Verify Rank 42
    is_valid_42, _ = _verify_decomposition(U42, V42, W42)
    
    if not is_valid_42:
        # Fallback to Schoolbook Rank 48 if Strassen fails (should not happen)
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
    
    # 3. Try to find Rank 41 if time permits
    # Only attempt if we have at least 30 seconds left
    if _elapsed() < TIME_LIMIT - 30:
        res_41 = _try_rank_41_als()
        if res_41 is not None:
            U41, V41, W41 = res_41
            # Final verification
            is_valid_41, _ = _verify_decomposition(U41, V41, W41)
            if is_valid_41:
                return {
                    "U": U41.tolist(),
                    "V": V41.tolist(),
                    "W": W41.tolist()
                }
    
    # 4. Return Rank 42
    return {"U": U42_list, "V": V42_list, "W": W42_list}
