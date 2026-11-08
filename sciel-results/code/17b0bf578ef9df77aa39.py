import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 3, 5]
DEADLINE = 140.0  # Safety margin

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give a valid decomposition of the multiplication tensor."""
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    # Efficient check: for each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check that non-tensor entries are zero
    T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
    if not np.allclose(T_recon, T, atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def _snap_to_half_integers(mat):
    """Snap values to nearest half-integer."""
    return np.round(mat * 2) / 2

def _try_rank(n, m, p, target_rank, time_limit=10):
    """Try to find a decomposition of target_rank using optimization."""
    start_time = time.time()
    T = _build_tensor(n, m, p)
    T_flat = T.flatten()
    
    def objective(params):
        U = params[:n*m*target_rank].reshape(n*m, target_rank)
        V = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W = params[n*m*target_rank + m*p*target_rank : ].reshape(p*n, target_rank)
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.sum((T_recon - T) ** 2)
    
    # Try a few random restarts
    for restart in range(5):
        if time.time() > start_time + time_limit:
            break
        # Initialize with random small values
        np.random.seed(restart)
        init_params = np.random.normal(0, 0.1, size=(n*m + m*p + p*n) * target_rank)
        
        # Use L-BFGS-B
        try:
            res = opt.minimize(objective, init_params, method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < 1e-6:
                # Extract solution
                U = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W = res.x[n*m*target_rank + m*p*target_rank : ].reshape(p*n, target_rank)
                
                # Snap to half-integers
                U_snapped = _snap_to_half_integers(U)
                V_snapped = _snap_to_half_integers(V)
                W_snapped = _snap_to_half_integers(W)
                
                if _verify(U_snapped, V_snapped, W_snapped, n, m, p):
                    return U_snapped, V_snapped, W_snapped
                else:
                    # Try without snapping if it's very close
                    if _verify(U, V, W, n, m, p):
                         return U, V, W
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = n * m * p
    
    # Try to improve
    for target_rank in range(best_rank - 1, 29, -1):
        if time.time() > DEADLINE - 10:
            break
        
        result = _try_rank(n, m, p, target_rank, time_limit=5)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break # Found a lower rank, stop searching lower ranks? 
                   # Actually, we might find an even lower one, but let's be safe and return first valid lower rank?
                   # The task says "rank-minimal". We should try to go as low as possible.
                   # But time is limited. Let's continue searching for lower ranks if time permits.
                   # However, finding rank 44 is already an improvement.
                   # Let's continue the loop to see if we can go lower.
                   # But the loop breaks if we find one? No, the loop continues.
                   # We should update best and continue.
    
    # Convert to list-of-lists
    def to_list(mat):
        result = []
        for row in mat:
            new_row = []
            for val in row:
                v = round(val * 2) / 2
                if abs(v - round(v)) < 1e-9:
                    new_row.append(int(round(v)))
                else:
                    new_row.append(v)
            result.append(new_row)
        return result
    
    U_out = to_list(best_U)
    V_out = to_list(best_V)
    W_out = to_list(best_W)
    
    # Final verification
    if not _verify(np.array(U_out, dtype=float), np.array(V_out, dtype=float), 
                   np.array(W_out, dtype=float), n, m, p):
        # Fall back to schoolbook
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        U_out = to_list(U_sb)
        V_out = to_list(V_sb)
        W_out = to_list(W_sb)
    
    return {"U": U_out, "V": V_out, "W": W_out}
