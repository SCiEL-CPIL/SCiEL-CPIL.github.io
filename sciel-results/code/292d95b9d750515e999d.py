import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor: T[i,j,k] = 1 for all valid (i,j,k)
    T = np.ones((n, m, p), dtype=float)
    
    # Verify a decomposition
    def verify(U, V, W, r, tol=1e-10):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    # Rank-55 decomposition: j=0 separately + Strassen for j=1,2
    def rank55_decomposition():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Part A: j=0 (first column of A, first row of B)
        for i in range(n):
            for k in range(p):
                U[i*m + 0, col] = 1
                V[0*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
        
        # Part B: j=1,2 with Strassen on 5 blocks of 2x2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1
            W[w_11, col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1
            V[v_22, col] += -1
            W[w_01, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1
            V[v_11, col] += -1
            W[w_00, col] += 1
            W[w_10, col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11, col] += 1
            U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1
            W[w_01, col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1
            U[u_11, col] += -1
            V[v_11, col] += 1
            V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1
            U[u_22, col] += -1
            V[v_21, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
        
        return U, V, W, rank
    
    U_best, V_best, W_best, rank_best = rank55_decomposition()
    err = verify(U_best, V_best, W_best, rank_best)
    
    # Fast verification using broadcasting
    def verify_fast(U, V, W, tol=1e-10):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Reshape to (n, m, r), (m, p, r), (p, n, r)
        r = U.shape[1]
        U_t = U.reshape(n, m, r)
        V_t = V.reshape(m, p, r)
        W_t = W.reshape(p, n, r)
        
        # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Product shape: (n, m, p, r)
        P = U_t[:, :, None, :] * V_t[None, :, :, :] * W_t[:, None, :, None]
        
        # Sum over r
        T_calc = np.sum(P, axis=-1)
        
        # Target is 1.0
        max_err = np.max(np.abs(T_calc - 1.0))
        return max_err

    # Randomized search for lower rank
    # Strategy:
    # 1. Randomize U and V.
    # 2. Solve for W using least squares on the flattened tensor.
    #    This is equivalent to solving for W[k*n+i, :] given U, V.
    #    The system is: sum_j U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    #    This is a linear system in W for fixed U, V.
    #    We can solve for all W simultaneously by forming a large matrix.
    
    def solve_for_W(U, V, r):
        # We need to find W such that T_calc = 1.
        # Let A be a matrix of size (n*p, r) where row (k*n+i) corresponds to W[k*n+i, :].
        # The constraint for each (i, j, k) is:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is not a simple linear system in W alone because W is multiplied by products of U and V.
        # However, for a fixed pair (i, k), the variable is W[k*n+i, :].
        # The equation is: sum_j (U[i*m+j, :] * V[j*p+k, :]) dot W[k*n+i, :] = 1
        # Let M_{i,k} be an m x r matrix with M_{i,k}[j, c] = U[i*m+j, c] * V[j*p+k, c].
        # Then M_{i,k} @ W[k*n+i, :]^T = ones(m).
        # This is a linear system for each (i, k).
        
        W = np.zeros((pn, r))
        for i in range(n):
            for k in range(p):
                M_ik = np.array([[U[i*m+j, c] * V[j*p+k, c] for c in range(r)] for j in range(m)])
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(M_ik, np.ones(m), rcond=None)
                    W[k*n+i, :] = sol
                except np.linalg.LinAlgError:
                    return None, 1e10
        return W, 0.0

    if time.time() < deadline - 10:
        # Try ranks from 54 down to 50
        for r in range(rank_best - 1, 49, -1):
            if time.time() > deadline - 5:
                break
            
            found = False
            # Increase trials as r decreases? No, keep it moderate to allow time for lower r.
            max_trials = 200
            
            for trial in range(max_trials):
                if time.time() > deadline - 5:
                    break
                
                np.random.seed(trial * 7919 + r * 1000)
                
                # Random initialization
                # Scaling is important. If U, V are too large, W will be small.
                # If too small, W will be large.
                # Let's start with standard normal.
                U_c = np.random.randn(nm, r)
                V_c = np.random.randn(mp, r)
                
                W_c, err_init = solve_for_W(U_c, V_c, r)
                if W_c is None:
                    continue
                
                # Verify
                max_err = verify_fast(U_c, V_c, W_c)
                
                # If we found a good solution, try to refine or accept
                if max_err < 1e-8:
                    # Try rounding to half-integers?
                    # The problem statement says "integer/half-integer entries".
                    # If the solution is close to half-integers, we should round.
                    U_r = np.round(U_c * 2) / 2
                    V_r = np.round(V_c * 2) / 2
                    W_r = np.round(W_c * 2) / 2
                    
                    max_err_rounded = verify_fast(U_r, V_r, W_r)
                    
                    if max_err_rounded < 1e-10:
                        U_best, V_best, W_best, rank_best = U_r, V_r, W_r, r
                        found = True
                        break
                    elif max_err < 1e-10:
                        # Accept unrounded if it's exact enough? 
                        # The contract says "integer/half-integer entries". 
                        # If we can't round, we might be invalid.
                        # However, if we find an exact solution that isn't half-integer, 
                        # it's not a valid answer per contract.
                        # But usually, if a low rank decomposition exists, it has integer/half-integer structure.
                        # Let's stick to rounded only for validity.
                        pass
            
            if found:
                break
    
    # Final verification
    final_err = verify(U_best, V_best, W_best, rank_best)
    if final_err > 1e-8:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        U_best, V_best, W_best = U, V, W
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
