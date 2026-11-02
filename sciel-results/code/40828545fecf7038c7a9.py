import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 4]

def _get_target_tensor(n, m, p):
    """Build the target multiplication tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            a = i*m + j
            for jp in range(m):
                for k in range(p):
                    b = jp*p + k
                    for ip in range(n):
                        for kp in range(p):
                            c = kp*n + ip
                            if i == ip and j == jp and k == kp:
                                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U,V,W exactly decompose the matrix multiplication tensor."""
    T_target = _get_target_tensor(n, m, p)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    max_err = np.max(np.abs(T_comp - T_target))
    return max_err < tol, max_err

def _build_rank14_strassen(n, m, p):
    """Build the known rank-14 Strassen decomposition for 2x2 * 2x4."""
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def set_col(col, u_dict, v_dict, w_dict):
        for idx, val in u_dict.items():
            U[idx, col] = val
        for idx, val in v_dict.items():
            V[idx, col] = val
        for idx, val in w_dict.items():
            W[idx, col] = val
            
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(0,3), 4=(1,0), 5=(1,1), 6=(1,2), 7=(1,3)
    # W indices (k*n+i): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1), 4=(2,0), 5=(2,1), 6=(3,0), 7=(3,1)
    
    # Left block (B[:, 0:2])
    set_col(0, {0:1, 3:1}, {0:1, 5:1}, {0:1, 3:1})
    set_col(1, {2:1, 3:1}, {0:1}, {1:1, 3:-1})
    set_col(2, {0:1}, {1:1, 5:-1}, {2:1, 3:1})
    set_col(3, {3:1}, {4:1, 0:-1}, {0:1, 1:1})
    set_col(4, {0:1, 1:1}, {5:1}, {0:-1, 2:1})
    set_col(5, {2:1, 0:-1}, {0:1, 1:1}, {3:1})
    set_col(6, {1:1, 3:-1}, {4:1, 5:1}, {0:1})
    
    # Right block (B[:, 2:4])
    set_col(7, {0:1, 3:1}, {2:1, 7:1}, {4:1, 7:1})
    set_col(8, {2:1, 3:1}, {2:1}, {5:1, 7:-1})
    set_col(9, {0:1}, {3:1, 7:-1}, {6:1, 7:1})
    set_col(10, {3:1}, {6:1, 2:-1}, {4:1, 5:1})
    set_col(11, {0:1, 1:1}, {7:1}, {4:-1, 6:1})
    set_col(12, {2:1, 0:-1}, {2:1, 3:1}, {7:1})
    set_col(13, {1:1, 3:-1}, {6:1, 7:1}, {4:1})
    
    return U, V, W

def _try_rank12(n, m, p, U14, V14, W14, deadline):
    """
    Try to find a rank-12 decomposition using nonlinear optimization.
    """
    T_target = _get_target_tensor(n, m, p)
    n_m = n * m  # 4
    m_p = m * p  # 8
    p_n = p * n  # 8
    rank = 12
    
    n_vars = n_m * rank + m_p * rank + p_n * rank
    
    def residuals(x):
        U = x[:n_m*rank].reshape(n_m, rank)
        V = x[n_m*rank:n_m*rank+m_p*rank].reshape(m_p, rank)
        W = x[n_m*rank+m_p*rank:].reshape(p_n, rank)
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        return (T_comp - T_target).flatten()
        
    # Pre-compute a random projection matrix P (14x12) to initialize
    # This helps the optimizer start near the "shape" of the rank 14 solution
    # but compressed to rank 12.
    
    best_err = float('inf')
    
    # Number of restarts
    # We have ~100s. Each LS call takes ~0.1-0.5s.
    # Let's try 150 restarts.
    n_restarts = 150
    
    for r in range(n_restarts):
        if time.time() > deadline - 2:
            break
        
        if r == 0:
            # Projection initialization
            P = np.random.randn(14, rank)
            U_init = U14 @ P
            V_init = V14 @ P
            W_init = W14 @ P
        elif r < 50:
            # Randomized projection
            P = np.random.randn(14, rank)
            # Add noise
            U_init = U14 @ P + np.random.randn(n_m, rank) * 0.1
            V_init = V14 @ P + np.random.randn(m_p, rank) * 0.1
            W_init = W14 @ P + np.random.randn(p_n, rank) * 0.1
        else:
            # Pure random initialization with varying scales
            scale = 0.5 ** (r // 50)
            U_init = np.random.randn(n_m, rank) * scale
            V_init = np.random.randn(m_p, rank) * scale
            W_init = np.random.randn(p_n, rank) * scale
            
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            # Use 'lm' for speed, 'trf' for robustness if needed
            # 'lm' is Levenberg-Marquardt, good for small unconstrained problems
            sol = least_squares(residuals, x0, method='lm', max_nfev=200, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            err = sol.cost
            
            if err < best_err:
                best_err = err
                
            if err < 1e-8:
                U = sol.x[:n_m*rank].reshape(n_m, rank)
                V = sol.x[n_m*rank:n_m*rank+m_p*rank].reshape(m_p, rank)
                W = sol.x[n_m*rank+m_p*rank:].reshape(p_n, rank)
                
                # Round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                valid, max_err = _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p)
                if valid:
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    
                # Try rounding to integers
                U_rounded_int = np.round(U)
                V_rounded_int = np.round(V)
                W_rounded_int = np.round(W)
                valid, max_err = _verify_decomposition(U_rounded_int, V_rounded_int, W_rounded_int, n, m, p)
                if valid:
                    return {"U": U_rounded_int.tolist(), "V": V_rounded_int.tolist(), "W": W_rounded_int.tolist()}
                    
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Build the known rank-14 solution
    U14, V14, W14 = _build_rank14_strassen(n, m, p)
    
    # Try to find a rank-12 solution
    result = _try_rank12(n, m, p, U14, V14, W14, deadline)
    if result is not None:
        return result
    
    # Fallback to rank 14
    return {"U": U14.tolist(), "V": V14.tolist(), "W": W14.tolist()}
