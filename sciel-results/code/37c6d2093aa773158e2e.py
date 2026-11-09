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
    
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()
    idx_v = (jj * p + kk).flatten()
    idx_w = (kk * n + ii).flatten()
    
    U_sel = U_np[idx_u, :]
    V_sel = V_np[idx_v, :]
    W_sel = W_np[idx_w, :]
    
    prod = U_sel * V_sel * W_sel
    vals = prod.sum(axis=1)
    
    err = np.abs(vals - 1.0)
    return float(err.max()) if err.size > 0 else 0.0

def _err_fast(U, V, W, idx_u, idx_v, idx_w):
    """Fast error computation."""
    U_sel = U[idx_u, :]
    V_sel = V[idx_v, :]
    W_sel = W[idx_w, :]
    vals = (U_sel * V_sel * W_sel).sum(axis=1)
    return float(np.max(np.abs(vals - 1.0)))

def _polish_decomposition(U, V, W, idx_u, idx_v, idx_w, deadline=None, max_iter=300):
    """High-precision polish pass with adaptive step size."""
    cur_err = _err_fast(U, V, W, idx_u, idx_v, idx_w)
    lr = 0.05
    patience = 0
    best_err = cur_err
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if deadline and time.time() > deadline - 5:
            break
        if cur_err < 1e-13:
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
        
        # Backtracking line search with more steps
        improved = False
        for alpha in [lr, lr*0.7, lr*0.5, lr*0.35, lr*0.25, lr*0.15, lr*0.1]:
            U_new = U - alpha * dU
            V_new = V - alpha * dV
            W_new = W - alpha * dW
            new_err = _err_fast(U_new, V_new, W_new, idx_u, idx_v, idx_w)
            if new_err < cur_err * 0.999:
                U, V, W = U_new, V_new, W_new
                cur_err = new_err
                improved = True
                if new_err < best_err:
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        
        if not improved:
            lr *= 0.5
            patience += 1
            if lr < 1e-10 or patience > 10:
                break
        else:
            patience = 0
            # Slightly increase lr if making good progress
            if cur_err < 0.1 * best_err:
                lr = min(lr * 1.1, 0.2)
        
        if it % 40 == 0 and it > 0:
            lr *= 0.9
    
    return best_U, best_V, best_W, best_err

def _try_rank_decomposition(target_rank, num_attempts=8, max_iter=500, deadline=None):
    """Try to find a rank decomposition using vectorized gradient descent."""
    n, m, p = DIMENSIONS
    
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    idx_u = (ii * m + jj).flatten()
    idx_v = (jj * p + kk).flatten()
    idx_w = (kk * n + ii).flatten()
    N = n * m * p
    
    for attempt in range(num_attempts):
        if deadline and time.time() > deadline - 15:
            return None
        
        np.random.seed(attempt * 7919 + target_rank * 31)
        
        if attempt == 0:
            # Random Gaussian
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
        elif attempt == 1:
            # Structured: cycle through (i,j,k)
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
            U += np.random.randn(n*m, target_rank) * 0.2
            V += np.random.randn(m*p, target_rank) * 0.2
            W += np.random.randn(p*n, target_rank) * 0.2
        elif attempt == 2:
            # Block-structured: group by i
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
            # Add structured noise
            for c in range(target_rank):
                i = c % n
                j = (c // n) % m
                U[i*m+j, c] += 0.5
                V[j*p, c] += 0.5
                W[(c % p)*n+i, c] += 0.5
            U += np.random.randn(n*m, target_rank) * 0.1
            V += np.random.randn(m*p, target_rank) * 0.1
            W += np.random.randn(p*n, target_rank) * 0.1
        elif attempt == 3:
            # Sparse random with higher values
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
            U += np.random.randn(n*m, target_rank) * 0.15
            V += np.random.randn(m*p, target_rank) * 0.15
            W += np.random.randn(p*n, target_rank) * 0.15
        elif attempt == 4:
            # Two-scale: mix structured and random
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            half = target_rank // 2
            for c in range(half):
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % p
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
            for c in range(half, target_rank):
                c2 = c - half
                i = c2 % n
                j = (c2 // n) % m
                k = (c2 // (n*m)) % p
                U[i*m+j, c] = 0.5
                V[j*p+k, c] = 0.5
                W[k*n+i, c] = 0.5
            U += np.random.randn(n*m, target_rank) * 0.1
            V += np.random.randn(m*p, target_rank) * 0.1
            W += np.random.randn(p*n, target_rank) * 0.1
        elif attempt == 5:
            # Diagonal-heavy initialization
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
                # Also add to neighboring positions
                if i+1 < n:
                    U[(i+1)*m+j, c] += 0.3
                if j+1 < m:
                    U[i*m+j+1, c] += 0.3
                if k+1 < p:
                    V[j*p+k+1, c] += 0.3
                if i+1 < n:
                    W[k*n+i+1, c] += 0.3
            U += np.random.randn(n*m, target_rank) * 0.1
            V += np.random.randn(m*p, target_rank) * 0.1
            W += np.random.randn(p*n, target_rank) * 0.1
        elif attempt == 6:
            # Random but scaled differently
            U = np.random.randn(n*m, target_rank) * 0.3
            V = np.random.randn(m*p, target_rank) * 0.3
            W = np.random.randn(p*n, target_rank) * 0.3
            # Add some structure
            for c in range(0, target_rank, 3):
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % p
                U[i*m+j, c] += 0.5
                V[j*p+k, c] += 0.5
                W[k*n+i, c] += 0.5
        else:
            # Pure random with different scale
            U = np.random.randn(n*m, target_rank) * 0.4
            V = np.random.randn(m*p, target_rank) * 0.4
            W = np.random.randn(p*n, target_rank) * 0.4
        
        lr = 0.03
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iter):
            if deadline and time.time() > deadline - 15:
                break
            
            U_sel = U[idx_u, :]
            V_sel = V[idx_v, :]
            W_sel = W[idx_w, :]
            
            prod = U_sel * V_sel * W_sel
            vals = prod.sum(axis=1)
            residuals = vals - 1.0
            
            VW = V_sel * W_sel
            UW = U_sel * W_sel
            UV = U_sel * V_sel
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            np.add.at(dU, idx_u, residuals[:, None] * VW)
            np.add.at(dV, idx_v, residuals[:, None] * UW)
            np.add.at(dW, idx_w, residuals[:, None] * UV)
            
            improved = False
            for alpha in [lr, lr*0.7, lr*0.5, lr*0.35, lr*0.25]:
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
                    improved = True
                    break
            
            if not improved:
                lr *= 0.5
                if lr < 1e-9:
                    break
            
            if best_err < 1e-10:
                break
            
            if iteration % 60 == 0 and iteration > 0:
                lr *= 0.88
        
        if best_err < 1e-3:
            U, V, W, final_err = _polish_decomposition(best_U, best_V, best_W, idx_u, idx_v, idx_w, deadline, max_iter=200)
            best_err = final_err
            best_U, best_V, best_W = U, V, W
        
        if best_err < 1e-7:
            U_rounded = np.round(best_U * 2) / 2
            V_rounded = np.round(best_V * 2) / 2
            W_rounded = np.round(best_W * 2) / 2
            
            err_rounded = _compute_error_vectorized(U_rounded, V_rounded, W_rounded)
            if err_rounded < 1e-9:
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Theoretical lower bound for matrix multiplication rank
    # For (n,m,p) = (4,5,6), known lower bounds suggest rank >= ~50-60
    # Try from 50 upward
    for target_rank in range(50, 120):
        if time.time() > deadline - 20:
            break
        
        # Give more attempts for smaller ranks (harder to find)
        if target_rank < 70:
            num_attempts = 6
            max_iter = 400
        else:
            num_attempts = 4
            max_iter = 350
        
        result = _try_rank_decomposition(target_rank, num_attempts=num_attempts, max_iter=max_iter, deadline=deadline)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
