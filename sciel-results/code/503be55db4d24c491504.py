import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We seek U (n*m, R), V (m*p, R), W (p*n, R) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify a decomposition
    def verify(U, V, W, tol=1e-8):
        """Check if sum_r U[:,r] outer V[:,r] outer W[:,r] == T (all ones) and entries are valid"""
        # Check if entries are integers or half-integers
        def is_half_int(A):
            return np.allclose(A, 2 * np.round(A / 2.0), atol=1e-8)
        
        if not (is_half_int(U) and is_half_int(V) and is_half_int(W)):
            return False, np.inf
            
        R = U.shape[1]
        T_comp = np.zeros((n, m, p))
        for r in range(R):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            for i in range(n):
                for k in range(p):
                    wk_i = w_r[k, i]
                    if wk_i == 0:
                        continue
                    T_comp[i, :, k] += u_r[i, :] * v_r[:, k] * wk_i
        
        err = np.max(np.abs(T_comp - 1.0))
        return err < tol, err
    
    # ALS for CP decomposition of the all-ones tensor
    def als_decompose(R, max_iters=200, tol=1e-10, seed=42):
        """
        Find U (n*m, R), V (m*p, R), W (p*n, R) such that
        sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        Returns (U, V, W, True) if a valid half-integer decomposition is found, else (None, None, None, False)
        """
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(n * m, R) * 0.5
        V = rng.randn(m * p, R) * 0.5
        W = rng.randn(p * n, R) * 0.5
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M_ij = np.zeros((p, R))
                    for k in range(p):
                        M_ij[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M_ij, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    N_jk = np.zeros((n, R))
                    for i in range(n):
                        N_jk[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(N_jk, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    P_ki = np.zeros((m, R))
                    for j in range(m):
                        P_ki[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(P_ki, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Snap to half-integers to maintain validity and check
            U_snapped = 2 * np.round(U / 2.0)
            V_snapped = 2 * np.round(V / 2.0)
            W_snapped = 2 * np.round(W / 2.0)
            
            # Update with snapped values for next iteration? 
            # Usually snapping during ALS can stall. 
            # Better: Check if snapped version works. If yes, return.
            # If no, keep floating point for next iteration but check convergence on floating point first?
            # The error requires integer/half-integer. So we MUST return snapped values.
            
            ok, err = verify(U_snapped, V_snapped, W_snapped)
            if ok:
                return U_snapped, V_snapped, W_snapped, True
                
            # Also check if the un-snapped version is close to a solution, 
            # but we can't return it. 
            # We continue iterating with the un-snapped values to potentially converge to a better region.
            # But we should probably nudge towards integers? 
            # For now, just iterate.
            
            # Check convergence of the floating point solution to detect stall
            # We need a floating point verification for stall detection
            T_comp = np.zeros((n, m, p))
            for r in range(R):
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                for i in range(n):
                    for k in range(p):
                        wk_i = w_r[k, i]
                        if wk_i == 0:
                            continue
                        T_comp[i, :, k] += u_r[i, :] * v_r[:, k] * wk_i
            fp_err = np.max(np.abs(T_comp - 1.0))
            
            if abs(prev_err - fp_err) < tol:
                break
            prev_err = fp_err
        
        # Final check with snapped values
        U_snapped = 2 * np.round(U / 2.0)
        V_snapped = 2 * np.round(V / 2.0)
        W_snapped = 2 * np.round(W / 2.0)
        ok, _ = verify(U_snapped, V_snapped, W_snapped)
        if ok:
            return U_snapped, V_snapped, W_snapped, True
        
        return None, None, None, False
    
    # Try various ranks from low to high
    # Known: schoolbook is 56. Literature suggests rank ~26-30 for 2x4x7.
    # Let's try ranks from 24 to 40
    
    best_result = None
    best_rank = n * m * p
    
    for R in range(24, 41):
        if time.time() > deadline - 10:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            U, V, W, ok = als_decompose(R, max_iters=100, seed=seed + R * 100)
            if ok:
                best_result = {"U": U, "V": V, "W": W}
                best_rank = R
                found = True
                break
        if found:
            break
    
    if best_result is not None:
        # Convert to lists
        return {
            "U": best_result["U"].tolist(),
            "V": best_result["V"].tolist(),
            "W": best_result["W"].tolist()
        }
    
    # Fallback to schoolbook
    return schoolbook()
