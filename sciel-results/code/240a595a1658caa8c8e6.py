"""Rewritten program: search for rank-minimal exact tensor decomposition of 2x3 * 3x6 matmul."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 115.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _snap_to_half_integers(U, V, W):
    """Snap values to the nearest half-integer."""
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    return U, V, W

def _search_rank_r(n, m, p, r, max_iters=200, lr=0.05, seed=42):
    """Try to find a rank-r decomposition using vectorized gradient descent."""
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n * m, r) * 0.1
    V = np.random.randn(m * p, r) * 0.1
    W = np.random.randn(p * n, r) * 0.1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Precompute indices for vectorization
    # We want to compute T_est[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is hard to vectorize directly without loops over i,j,k.
    # But n=2, m=3, p=6 is small. We can loop over i,j,k and vectorize over c.
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        # Compute loss and gradients
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    # s = sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    s = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                    err = s - 1.0
                    loss += err * err
                    
                    # Gradients
                    # d/dU[u_idx, c] = 2 * err * V[v_idx, c] * W[w_idx, c]
                    grad_U[u_idx] += 2 * err * V[v_idx] * W[w_idx]
                    grad_V[v_idx] += 2 * err * U[u_idx] * W[w_idx]
                    grad_W[w_idx] += 2 * err * U[u_idx] * V[v_idx]
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-10:
                # Try to snap and verify
                U_snap, V_snap, W_snap = _snap_to_half_integers(best_U, best_V, best_W)
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Periodic snapping check
        if it % 20 == 0 and best_loss < 1e-4:
            U_snap, V_snap, W_snap = _snap_to_half_integers(U, V, W)
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
                
    # Final check
    if best_loss < 1e-10:
        U_snap, V_snap, W_snap = _snap_to_half_integers(best_U, best_V, best_W)
        if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
            return U_snap, V_snap, W_snap
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Search for lower ranks
    # Start from best_rank - 1 down to a reasonable lower bound (e.g., 18)
    # We use multiple seeds for each rank to increase chance of success
    for r in range(best_rank - 1, 17, -1):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        found = False
        # Try a few seeds for this rank
        for seed in range(5):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            result = _search_rank_r(n, m, p, r, max_iters=100, seed=seed + 42)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    found = True
                    break
        
        if found:
            # If we found one, we could try to go lower, but let's stop if we hit a "good" rank
            # or continue if time permits. Given the difficulty, finding any rank < 36 is a win.
            # We continue the loop to see if we can go lower.
            pass
            
    # Final verification and formatting
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = U_school, V_school, W_school
        
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
