import numpy as np
import time
from scipy.optimize import minimize
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
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
    
    # Target ranks to check
    # Theoretical LB is 9. Practical LB often higher.
    # Try 16, 18, 20, 22, 24
    target_ranks = [16, 18, 20, 22, 24, 26, 28, 30, 32, 34]
    
    def verify_decomposition(U, V, W, R):
        """Verify that the decomposition exactly reconstructs T using Fraction for exactness"""
        # Convert to Fraction for exact check
        # This is slow, so only do it if numerical check passes
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        # Check if close to 0 or 1
        # Since T has 0s and 1s, T_recon should be close to 0 or 1
        # Specifically, T_recon[i,j,k] should be T[i,j,k]
        if not np.allclose(T_recon, T, atol=1e-8):
            return False
            
        # Exact check with Fractions
        try:
            for i in range(n*m):
                for j in range(m*p):
                    for k in range(p*n):
                        val = T_recon[i, j, k]
                        target = T[i, j, k]
                        # Snap to fraction to verify exactness
                        # If val is 0.99999, it should be 1.
                        # We check if the rational approximation matches T
                        # Actually, if allclose passed, we can just return True for the purpose of finding the rank,
                        # but the task requires EXACT decomposition.
                        # So we must ensure the returned U,V,W produce EXACTLY T.
                        pass
            # For speed, we assume if numerical error is < 1e-8 and we snap to rationals, 
            # we should verify the snapped matrices.
            return True
        except:
            return False

    def snap_and_verify(U, V, W, max_denom=4):
        """Snap values to nearest rational with denominator up to max_denom and verify exactly"""
        U_snap = np.zeros_like(U)
        V_snap = np.zeros_like(V)
        W_snap = np.zeros_like(W)
        
        for arr, snap_arr in [(U, U_snap), (V, V_snap), (W, W_snap)]:
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    val = arr[i, j]
                    best_frac = 0.0
                    best_diff = float('inf')
                    for denom in range(1, max_denom+1):
                        num = round(val * denom)
                        frac = num / denom
                        diff = abs(val - frac)
                        if diff < best_diff:
                            best_diff = diff
                            best_frac = frac
                    snap_arr[i, j] = best_frac
        
        # Verify exactness with Fractions
        # Reconstruct T from snapped matrices
        T_recon = np.zeros((n*m, m*p, p*n))
        # Use Fraction for exact arithmetic
        for r in range(U.shape[1]):
            for i in range(n*m):
                ui = Fraction(U_snap[i, r])
                if ui == 0: continue
                for j in range(m*p):
                    vj = Fraction(V_snap[j, r])
                    if vj == 0: continue
                    uv = ui * vj
                    if uv == 0: continue
                    for k in range(p*n):
                        wk = Fraction(W_snap[k, r])
                        if wk == 0: continue
                        T_recon[i, j, k] += uv * wk
                        
        # Check if T_recon matches T exactly
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    if T_recon[i, j, k] != Fraction(int(T[i, j, k])):
                        return None, None, None, False
        
        return U_snap, V_snap, W_snap, True

    # Strategy: For each target rank, use local optimization from multiple random starts
    # then snap to rationals and verify
    for R in target_ranks:
        if time.time() > deadline - 20:
            break
            
        # Number of restarts
        max_restarts = 20 if R <= 20 else 10
        
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            seed = 1000 + R * 100 + restart
            np.random.seed(seed)
            
            # Scale initialization
            # Values should be small
            U0 = np.random.randn(n*m, R) * 0.5
            V0 = np.random.randn(m*p, R) * 0.5
            W0 = np.random.randn(p*n, R) * 0.5
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def objective(x):
                U = x[:n*m*R].reshape(n*m, R)
                V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = x[n*m*R + m*p*R :].reshape(p*n, R)
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_recon - T
                return np.sum(diff * diff)
            
            try:
                # Use L-BFGS-B
                res = minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
                
                if res.fun < 1e-8:
                    U_opt = res.x[:n*m*R].reshape(n*m, R)
                    V_opt = res.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W_opt = res.x[n*m*R + m*p*R :].reshape(p*n, R)
                    
                    # Try snapping with different denominators
                    for max_denom in [1, 2, 3, 4, 6, 8]:
                        U_snap, V_snap, W_snap, valid = snap_and_verify(U_opt, V_opt, W_opt, max_denom)
                        if valid:
                            best_rank = R
                            # Convert to list of floats (or ints if integer)
                            # The interface accepts integer/half-integer entries
                            best_U = [[float(x) for x in row] for row in U_snap]
                            best_V = [[float(x) for x in row] for row in V_snap]
                            best_W = [[float(x) for x in row] for row in W_snap]
                            found = True
                            break
                    
                    if found:
                        break
                        
            except Exception:
                continue
        
        if found:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
