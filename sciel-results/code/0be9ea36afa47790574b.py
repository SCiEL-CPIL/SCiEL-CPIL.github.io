import numpy as np
import time
from scipy.optimize import minimize
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute the target tensor T
    # T[u_idx, v_idx, w_idx] = 1 where u_idx = i*m+j, v_idx = j*p+k, w_idx = k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Initialize schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_sb
    
    # Try ranks from low to high
    # For 3x3x4 matrix multiplication, known results suggest rank could be around 18-24
    target_ranks = [18, 20, 22, 24, 26, 28]
    
    def verify_decomposition(U, V, W, R):
        """Verify that the decomposition exactly reconstructs T"""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-10)
    
    def snap_to_rationals(U, V, W, max_denom=4):
        """Snap values to nearest rational with denominator up to max_denom"""
        U_snap = np.zeros_like(U)
        V_snap = np.zeros_like(V)
        W_snap = np.zeros_like(W)
        
        for arr, snap_arr in [(U, U_snap), (V, V_snap), (W, W_snap)]:
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    val = arr[i, j]
                    # Try to snap to rational with small denominator
                    best_frac = None
                    best_diff = float('inf')
                    for denom in range(1, max_denom+1):
                        num = round(val * denom)
                        frac = num / denom
                        diff = abs(val - frac)
                        if diff < best_diff:
                            best_diff = diff
                            best_frac = frac
                    snap_arr[i, j] = best_frac
        
        return U_snap, V_snap, W_snap
    
    # Strategy: For each target rank, use local optimization from multiple random starts
    # then snap to rationals and verify
    for R in target_ranks:
        if time.time() > deadline - 30:
            break
            
        # Number of restarts based on rank
        max_restarts = 8 if R <= 22 else 5
        
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Random initialization with different seeds
            seed = 100 + R * 10 + restart
            np.random.seed(seed)
            
            # Initialize with small random values
            U0 = np.random.randn(n*m, R) * 0.5
            V0 = np.random.randn(m*p, R) * 0.5
            W0 = np.random.randn(p*n, R) * 0.5
            
            # Flatten parameters
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def objective(x):
                U = x[:n*m*R].reshape(n*m, R)
                V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = x[n*m*R + m*p*R :].reshape(p*n, R)
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_recon - T
                return np.sum(diff * diff)
            
            # Use L-BFGS-B optimization
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < 1e-8:
                    U_opt = res.x[:n*m*R].reshape(n*m, R)
                    V_opt = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W_opt = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                    
                    # Try different snap strategies
                    for max_denom in [2, 3, 4, 6, 8]:
                        U_snap, V_snap, W_snap = snap_to_rationals(U_opt, V_opt, W_opt, max_denom)
                        
                        if verify_decomposition(U_snap, V_snap, W_snap, R):
                            best_rank = R
                            best_U = U_snap.tolist()
                            best_V = V_snap.tolist()
                            best_W = W_snap.tolist()
                            found = True
                            break
                    
                    if found:
                        break
                        
            except Exception:
                continue
        
        if found:
            # Found a valid decomposition at rank R
            # Since we iterate from low to high, this is the minimal rank found
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
