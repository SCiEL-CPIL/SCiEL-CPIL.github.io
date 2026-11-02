import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def _snap_to_half_int(arr):
    """Snap array values to nearest half-integer."""
    return np.round(arr * 2) / 2

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify if U, V, W exactly decompose the matrix multiplication tensor."""
    # Reconstruct tensor
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    R = U.shape[1]
    # Use einsum for efficiency
    # U_idx: i*m+j
    # V_idx: j*p+k
    # W_idx: k*n+i
    
    # Reshape U to (n, m, R)
    U_t = U.reshape(n, m, R)
    # Reshape V to (m, p, R)
    V_t = V.reshape(m, p, R)
    # Reshape W to (p, n, R)
    W_t = W.reshape(p, n, R)
    
    # Sum over r
    # T[i, j, k] = sum_r U_t[i, j, r] * V_t[j, k, r] * W_t[k, i, r]
    # Note: V_t indices are (j, k, r) but V is defined as V[j*p+k, r].
    # So V_t[j, k, r] is correct.
    # W_t[k, i, r] is correct.
    
    # We need to sum over r.
    # T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    # Expected tensor is 1 for all i,j,k
    expected = np.ones((n, m, p))
    
    return np.allclose(T, expected, atol=tol)

def _find_decomposition(n, m, p, rank, time_limit=10):
    """Find a rank decomposition for M_{n,m,p} using optimization."""
    start_time = time.time()
    
    # Target tensor
    # T[i, j, k] = 1
    
    # Initialize parameters
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    # Number of restarts based on time
    max_restarts = 20
    
    for r in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        # Scale matters. Let's try small random values.
        scale = 1.0
        U = np.random.randn(n * m, rank) * scale
        V = np.random.randn(m * p, rank) * scale
        W = np.random.randn(p * n, rank) * scale
        
        # Flatten
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def residual(x):
            U_flat = x[:n*m*rank].reshape(n*m, rank)
            V_flat = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_flat = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            U_t = U_flat.reshape(n, m, rank)
            V_t = V_flat.reshape(m, p, rank)
            W_t = W_flat.reshape(p, n, rank)
            
            T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            # Residual is T - 1
            return (T - 1).flatten()
        
        try:
            res = least_squares(residual, x0, method='lm', max_nfev=1000)
            if res.success or res.cost < best_err:
                x_opt = res.x
                U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
                V_opt = x_opt[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_opt = x_opt[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Snap to half-integers
                U_snap = _snap_to_half_int(U_opt)
                V_snap = _snap_to_half_int(V_opt)
                W_snap = _snap_to_half_int(W_opt)
                
                # Verify snapped solution
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap
                
                # If snapped fails, check unsnapped error
                if res.cost < best_err:
                    best_err = res.cost
                    best_U = U_opt
                    best_V = V_opt
                    best_W = W_opt
                    
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find better decomposition
    # Hypothesis: Rank of M_{2,3,2} is 11.
    # If we find rank 11 for 2x3x2, we can build rank 22 for 2x3x4.
    
    if n == 2 and m == 3 and p == 4:
        # Try to solve for 2x3x2
        n_sub, m_sub, p_sub = 2, 3, 2
        rank_sub = 11
        
        U_sub, V_sub, W_sub = _find_decomposition(n_sub, m_sub, p_sub, rank_sub, time_limit=30)
        
        if U_sub is not None:
            # Construct 2x3x4 decomposition
            # Rank = 2 * rank_sub = 22
            rank_total = 2 * rank_sub
            U_total = np.zeros((n * m, rank_total))
            V_total = np.zeros((m * p, rank_total))
            W_total = np.zeros((p * n, rank_total))
            
            # Copy 1: k=0,1
            U_total[:, :rank_sub] = U_sub
            # V_sub is (m*p_sub, rank_sub) = (6, 11)
            # V_total is (m*p, rank_total) = (12, 22)
            # V_total[0:6, :11] = V_sub
            V_total[:m*p_sub, :rank_sub] = V_sub
            # W_sub is (p_sub*n, rank_sub) = (4, 11)
            # W_total is (p*n, rank_total) = (8, 22)
            # W_total[0:4, :11] = W_sub
            W_total[:p_sub*n, :rank_sub] = W_sub
            
            # Copy 2: k=2,3
            U_total[:, rank_sub:] = U_sub
            # V_total[6:12, 11:22] = V_sub
            V_total[m*p_sub:, rank_sub:] = V_sub
            # W_total[4:8, 11:22] = W_sub
            W_total[p_sub*n:, rank_sub:] = W_sub
            
            # Verify
            if _verify_decomposition(U_total, V_total, W_total, n, m, p):
                # Convert to lists of lists for output
                return {
                    "U": U_total.tolist(),
                    "V": V_total.tolist(),
                    "W": W_total.tolist()
                }
                
    return fallback

