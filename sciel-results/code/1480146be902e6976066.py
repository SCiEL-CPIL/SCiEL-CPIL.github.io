import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor
    M = np.zeros((n*m, m*p, p*n))
    nz_idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx1 = i*m + j
                idx2 = j*p + k
                idx3 = k*n + i
                M[idx1, idx2, idx3] = 1.0
                nz_idx.append((idx1, idx2, idx3))
    nz_idx = np.array(nz_idx)
    
    # Precompute flattened matrices for ALS
    M_flat = M.reshape(n*m, m*p * p*n)
    M_T_flat = M.T.reshape(m*p, n*m*p*n)
    M_P_flat = M.reshape(n*m*m*p, p*n).T
    
    # Schoolbook fallback
    def make_schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = n * m * p
    
    lower_bound = max(n*p, m*n, m*p)
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly matches the tensor."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx1 = i*m + j
                    idx2 = j*p + k
                    idx3 = k*n + i
                    s = np.sum(U_np[idx1] * V_np[idx2] * W_np[idx3])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def run_als(rank, seed, max_iters=80, tol=1e-10):
        """Run ALS for a given rank and seed. Returns (U, V, W, residual) or None."""
        np.random.seed(seed)
        
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        prev_residual = np.inf
        stagnant_count = 0
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            T = (V[:, np.newaxis, :] * W[np.newaxis, :, :])
            T_flat = T.reshape(m*p*p*n, rank)
            try:
                U_new, _, _, _ = np.linalg.lstsq(T_flat.T, M_flat.T, rcond=None)
                U_new = U_new.T
            except:
                break
            
            # Update V
            T2 = (U[:, np.newaxis, :] * W[np.newaxis, :, :])
            T2_flat = T2.reshape(n*m*p*n, rank)
            try:
                V_new, _, _, _ = np.linalg.lstsq(T2_flat.T, M_T_flat.T, rcond=None)
                V_new = V_new.T
            except:
                break
            
            # Update W
            T3 = (U[:, np.newaxis, :] * V[np.newaxis, :, :])
            T3_flat = T3.reshape(n*m*m*p, rank)
            try:
                W_new, _, _, _ = np.linalg.lstsq(T3_flat.T, M_P_flat.T, rcond=None)
                W_new = W_new.T
            except:
                break
            
            U, V, W = U_new, V_new, W_new
            
            # Compute residual
            i_vals = nz_idx[:, 0]
            j_vals = nz_idx[:, 1]
            k_vals = nz_idx[:, 2]
            U_vals = U[i_vals, :]
            V_vals = V[j_vals, :]
            W_vals = W[k_vals, :]
            tensor_vals = np.sum(U_vals * V_vals * W_vals, axis=1)
            residual = np.abs(tensor_vals - 1.0).max()
            
            if residual < tol:
                return U, V, W, residual
            
            # Detect stagnation
            if residual > prev_residual * 0.995:
                stagnant_count += 1
            else:
                stagnant_count = 0
            prev_residual = residual
            
            if stagnant_count > 15:
                break
        
        return U, V, W, residual
    
    def polish(U, V, W, rank, max_iters=20, tol=1e-12):
        """High-precision polish pass."""
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            T = (V[:, np.newaxis, :] * W[np.newaxis, :, :])
            T_flat = T.reshape(m*p*p*n, rank)
            try:
                U_new, _, _, _ = np.linalg.lstsq(T_flat.T, M_flat.T, rcond=None)
                U_new = U_new.T
            except:
                break
            
            T2 = (U[:, np.newaxis, :] * W[np.newaxis, :, :])
            T2_flat = T2.reshape(n*m*p*n, rank)
            try:
                V_new, _, _, _ = np.linalg.lstsq(T2_flat.T, M_T_flat.T, rcond=None)
                V_new = V_new.T
            except:
                break
            
            T3 = (U[:, np.newaxis, :] * V[np.newaxis, :, :])
            T3_flat = T3.reshape(n*m*m*p, rank)
            try:
                W_new, _, _, _ = np.linalg.lstsq(T3_flat.T, M_P_flat.T, rcond=None)
                W_new = W_new.T
            except:
                break
            
            U, V, W = U_new, V_new, W_new
            
            i_vals = nz_idx[:, 0]
            j_vals = nz_idx[:, 1]
            k_vals = nz_idx[:, 2]
            U_vals = U[i_vals, :]
            V_vals = V[j_vals, :]
            W_vals = W[k_vals, :]
            tensor_vals = np.sum(U_vals * V_vals * W_vals, axis=1)
            residual = np.abs(tensor_vals - 1.0).max()
            
            if residual < tol:
                break
        
        return U, V, W
    
    # Search for minimal rank
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 15:
            break
        
        found = False
        max_restarts = 30 if rank <= 30 else 15
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            seed = rank * 10000 + restart * 7 + 42
            
            result = run_als(rank, seed)
            if result is None:
                continue
            
            U, V, W, residual = result
            
            if residual < 1e-9:
                # Polish for higher precision
                U, V, W = polish(U, V, W, rank)
                
                # Final verification
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                if verify_decomposition(U_list, V_list, W_list, tol=1e-8):
                    best_rank = rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break
        
        if found:
            break
    
    return best_result
