"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook():
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

def _compute_error_vectorized(U, V, W):
    """Compute the max error using vectorized operations."""
    n, m, p = DIMENSIONS
    U_np = np.asarray(U, dtype=float)
    V_np = np.asarray(V, dtype=float)
    W_np = np.asarray(W, dtype=float)
    r = U_np.shape[1]
    
    # Build index arrays
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = ii * m + jj
    idx_v = jj * p + kk
    idx_w = kk * n + ii
    
    # For each (i,j,k), compute sum_c U[idx_u,c] * V[idx_v,c] * W[idx_w,c]
    # U[idx_u, :] has shape (n*m*p, r)
    U_sel = U_np[idx_u.flatten(), :]  # (n*m*p, r)
    V_sel = V_np[idx_v.flatten(), :]  # (n*m*p, r)
    W_sel = W_np[idx_w.flatten(), :]  # (n*m*p, r)
    
    # Element-wise product then sum over c
    prod = U_sel * V_sel * W_sel  # (n*m*p, r)
    vals = prod.sum(axis=1)  # (n*m*p,)
    
    err = np.abs(vals - 1.0)
    return float(err.max()) if err.size > 0 else 0.0

def _polish_decomposition(U, V, W, deadline=None, max_iter=200):
    """High-precision polish pass with backtracking line search."""
    n, m, p = DIMENSIONS
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()
    idx_v = (jj * p + kk).flatten()
    idx_w = (kk * n + ii).flatten()
    
    def _err(U, V, W):
        U_sel = U[idx_u, :]
        V_sel = V[idx_v, :]
        W_sel = W[idx_w, :]
        vals = (U_sel * V_sel * W_sel).sum(axis=1)
        return float(np.max(np.abs(vals - 1.0)))
    
    cur_err = _err(U, V, W)
    lr = 0.1
    for it in range(max_iter):
        if deadline and time.time() > deadline - 5:
            break
        if cur_err < 1e-12:
            break
        
        U_sel = U[idx_u, :]
        V_sel = V[idx_v, :]
        W_sel = W[idx_w, :]
        vals = (U_sel * V_sel * W_sel).sum(axis=1)
        res = vals - 1.0
        
        VW = V_sel * W_sel
        UW = U_sel * W_sel
        UV = U_sel * V_sel
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        np.add.at(dU, idx_u, res[:, None] * VW)
        np.add.at(dV, idx_v, res[:, None] * UW)
        np.add.at(dW, idx_w, res[:, None] * UV)
        
        # Backtracking line search
        for alpha in [lr, lr*0.5, lr*0.25, lr*0.125, lr*0.0625]:
            U_new = U - alpha * dU
            V_new = V - alpha * dV
            W_new = W - alpha * dW
            new_err = _err(U_new, V_new, W_new)
            if new_err < cur_err:
                U, V, W = U_new, V_new, W_new
                cur_err = new_err
                break
        else:
            lr *= 0.5
            if lr < 1e-8:
                break
        
        if it % 30 == 0 and it > 0:
            lr *= 0.9
    
    return U, V, W, cur_err

def _try_rank_decomposition(target_rank, num_attempts=5, max_iter=400, deadline=None):
    """Try to find a rank decomposition using vectorized gradient descent."""
    n, m, p = DIMENSIONS
    
    # Precompute index arrays
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()  # (n*m*p,)
    idx_v = (jj * p + kk).flatten()  # (n*m*p,)
    idx_w = (kk * n + ii).flatten()  # (n*m*p,)
    N = n * m * p  # 120
    
    for attempt in range(num_attempts):
        if deadline and time.time() > deadline - 10:
            return None
        
        np.random.seed(attempt * 7919 + target_rank)
        
        # Multiple initialization strategies
        if attempt % 3 == 0:
            # Standard random
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
        elif attempt % 3 == 1:
            # Sparse initialization with some 1s
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            for c in range(target_rank):
                ui = np.random.randint(0, n*m)
                vi = np.random.randint(0, m*p)
                wi = np.random.randint(0, p*n)
                U[ui, c] = 1.0
                V[vi, c] = 1.0
                W[wi, c] = 1.0
            U += np.random.randn(n*m, target_rank) * 0.1
            V += np.random.randn(m*p, target_rank) * 0.1
            W += np.random.randn(p*n, target_rank) * 0.1
        else:
            # Structured: try to hit the right indices more often
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            for c in range(target_rank):
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % p
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
            U += np.random.randn(n*m, target_rank) * 0.3
            V += np.random.randn(m*p, target_rank) * 0.3
            W += np.random.randn(p*n, target_rank) * 0.3
        
        lr = 0.05
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iter):
            if deadline and time.time() > deadline - 10:
                break
            
            # Compute residuals: for each (i,j,k), err_ijk = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1
            U_sel = U[idx_u, :]  # (N, r)
            V_sel = V[idx_v, :]  # (N, r)
            W_sel = W[idx_w, :]  # (N, r)
            
            prod = U_sel * V_sel * W_sel  # (N, r)
            vals = prod.sum(axis=1)  # (N,)
            residuals = vals - 1.0  # (N,)
            
            # Gradients using vectorized operations
            VW = V_sel * W_sel  # (N, r)
            UW = U_sel * W_sel  # (N, r)
            UV = U_sel * V_sel  # (N, r)
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            np.add.at(dU, idx_u, residuals[:, None] * VW)
            np.add.at(dV, idx_v, residuals[:, None] * UW)
            np.add.at(dW, idx_w, residuals[:, None] * UV)
            
            # Backtracking line search
            for alpha in [lr, lr*0.5, lr*0.25, lr*0.125]:
                U_new = U - alpha * dU
                V_new = V - alpha * dV
                W_new = W - alpha * dW
                
                U_sel_new = U_new[idx_u, :]
                V_sel_new = V_new[idx_v, :]
                W_sel_new = W_new[idx_w, :]
                new_vals = (U_sel_new * V_sel_new * W_sel_new).sum(axis=1)
                new_err = float(np.max(np.abs(new_vals - 1.0)))
                
                if new_err < best_err:
                    U, V, W = U_new, V_new, W_new
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            else:
                lr *= 0.5
                if lr < 1e-8:
                    break
            
            if best_err < 1e-10:
                break
            
            # Decay learning rate
            if iteration % 50 == 0 and iteration > 0:
                lr *= 0.85
        
        # Final polish pass
        if best_err < 1e-4:
            U, V, W, final_err = _polish_decomposition(best_U, best_V, best_W, deadline, max_iter=150)
            best_err = final_err
            best_U, best_V, best_W = U, V, W
        
        if best_err < 1e-6:
            # Try to round to integers/half-integers
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            err_rounded = _compute_error_vectorized(U_rounded, V_rounded, W_rounded)
            if err_rounded < 1e-10:
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
            
            # If direct rounding fails, try local search around the rounded values
            # This is expensive, so only do it if we are very close
            if best_err < 1e-5 and deadline and time.time() < deadline - 5:
                 # Simple local search: try flipping signs or adding 0.5 to random entries
                 # This is a heuristic
                 pass

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    n, m, p = DIMENSIONS
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try to find lower-rank decompositions
    # Try ranks from 60 up to 119, looking for the smallest that works
    for target_rank in range(60, 120):
        if time.time() > deadline - 30:
            break
        
        result = _try_rank_decomposition(target_rank, num_attempts=3, max_iter=300, deadline=deadline)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
