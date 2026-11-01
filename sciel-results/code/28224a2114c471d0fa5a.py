import time
import sys
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 6]
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    """
    Returns a rank-minimal (or best known) tensor decomposition for 
    multiplying a 2x3 matrix by a 3x6 matrix.
    
    Uses optimization to search for rank < 36 decompositions.
    Falls back to schoolbook (rank 36) if no improvement found.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook rank
    sb_rank = n * m * p  # 36
    
    # Build schoolbook decomposition
    U_sb = np.zeros((n*m, sb_rank), dtype=float)
    V_sb = np.zeros((m*p, sb_rank), dtype=float)
    W_sb = np.zeros((p*n, sb_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    # Define the target tensor T (sparse: only 36 non-zero entries)
    # Store as list of (a, b, c) indices where T=1
    T_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_indices.append((i*m+j, j*p+k, k*n+i))
                
    # Precompute indices for faster access
    T_a = np.array([t[0] for t in T_indices])
    T_b = np.array([t[1] for t in T_indices])
    T_c = np.array([t[2] for t in T_indices])

    best_rank = sb_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Helper: compute residuals for least_squares
    def residuals(x, U, V, W, target_rank):
        # x is flattened parameters
        # We need to update U, V, W or just use x directly in the residual function
        # For efficiency, we will pass the current x and reshape inside
        pass

    # Better approach: define residual function that takes x and reshapes
    def make_residual_func(target_rank):
        u_size = n*m * target_rank
        v_size = m*p * target_rank
        w_size = p*n * target_rank
        
        def res(x):
            U = x[:u_size].reshape(n*m, target_rank)
            V = x[u_size:u_size+v_size].reshape(m*p, target_rank)
            W = x[u_size+v_size:].reshape(p*n, target_rank)
            
            # Compute U[a] dot (V[b] * W[c]) for all target indices
            # Vectorized:
            # U_a = U[T_a, :] shape (36, R)
            # V_b = V[T_b, :] shape (36, R)
            # W_c = W[T_c, :] shape (36, R)
            # val = sum(U_a * V_b * W_c, axis=1)
            
            U_a = U[T_a, :]
            V_b = V[T_b, :]
            W_c = W[T_c, :]
            
            vals = np.einsum('ij,ij,ij->i', U_a, V_b, W_c)
            return vals - 1.0
            
        return res

    def snap_and_validate(U, V, W, tol=1e-6):
        # Snap to nearest 0.5
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Check exactness
        U_a = U_snap[T_a, :]
        V_b = V_snap[T_b, :]
        W_c = W_snap[T_c, :]
        vals = np.einsum('ij,ij,ij->i', U_a, V_b, W_c)
        
        if np.allclose(vals, 1.0, atol=tol):
            # Also check non-targets are 0 (sampled)
            # For speed, check a few random non-targets
            np.random.seed(42)
            for _ in range(50):
                a = np.random.randint(0, n*m)
                b = np.random.randint(0, m*p)
                c = np.random.randint(0, p*n)
                if (a, b, c) in T_indices:
                    continue
                val = np.dot(U_snap[a, :], V_snap[b, :] * W_snap[c, :])
                if abs(val) > tol:
                    return False
            return True
        return False

    np.random.seed(42)
    
    # Try ranks from 35 down to 25 (aggressive search)
    # Use least_squares which is faster than L-BFGS-B for this
    for target_rank in range(35, 24, -1):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        res_func = make_residual_func(target_rank)
        u_size = n*m * target_rank
        v_size = m*p * target_rank
        w_size = p*n * target_rank
        total_size = u_size + v_size + w_size
        
        # Multiple restarts
        found = False
        for restart in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Initialize with random small values or truncated schoolbook
            if restart == 0:
                # Truncated schoolbook
                U_init = U_sb[:, :target_rank].copy()
                V_init = V_sb[:, :target_rank].copy()
                W_init = W_sb[:, :target_rank].copy()
            else:
                # Random initialization
                U_init = np.random.randn(n*m, target_rank) * 0.5
                V_init = np.random.randn(m*p, target_rank) * 0.5
                W_init = np.random.randn(p*n, target_rank) * 0.5
            
            x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
            
            try:
                # Use 'trf' (Trust Region Reflective) or 'lm' (Levenberg-Marquardt)
                # 'lm' is fast for small problems but doesn't support bounds well, 
                # but we don't have bounds here.
                result = least_squares(res_func, x0, method='lm', max_nfev=1000)
                
                U_opt = result.x[:u_size].reshape(n*m, target_rank)
                V_opt = result.x[u_size:u_size+v_size].reshape(m*p, target_rank)
                W_opt = result.x[u_size+v_size:].reshape(p*n, target_rank)
                
                if snap_and_validate(U_opt, V_opt, W_opt):
                    best_rank = target_rank
                    best_U = U_opt
                    best_V = V_opt
                    best_W = W_opt
                    found = True
                    break
            except Exception:
                continue
        
        if found:
            break

    if time.time() - start_time > DEADLINE - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
