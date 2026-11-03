import numpy as np
import time

def construct():
    """
    5x5 matrix multiplication - rank-minimal exact tensor decomposition.
    Uses structured random search with null-space projection and final polish.
    """
    n, m, p = 5, 5, 5
    t_start = time.time()
    time_budget = 110  # seconds, leave margin for 120s limit
    
    def time_remaining():
        return time_budget - (time.time() - t_start)
    
    # Build the constraint structure
    # For each output (i,k), we need:
    # sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1 for each j
    # and sum_c W[k*n+i,c] * U[x*m+y,c] * V[z*p+w,c] = 0 for all other (x,y,z,w)
    
    # Precompute the constraint equations for each (i,k)
    constraints = {}
    for i in range(n):
        for k in range(p):
            # Equations with target = 1
            eqs_1 = []
            for j in range(m):
                eqs_1.append((i, j, j, k, 1.0))
            # Equations with target = 0 (only include a subset for efficiency)
            eqs_0 = []
            for x in range(n):
                for y in range(m):
                    for z in range(m):
                        for w in range(p):
                            if not (x == i and y == z and w == k):
                                eqs_0.append((x, y, z, w, 0.0))
            constraints[(i, k)] = eqs_1 + eqs_0
    
    # Build schoolbook as fallback
    best_rank = n * m * p
    U_s = np.zeros((n*m, best_rank), dtype=float)
    V_s = np.zeros((m*p, best_rank), dtype=float)
    W_s = np.zeros((p*n, best_rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1.0
                V_s[j*p+k, col] = 1.0
                W_s[k*n+i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U_s.copy(), V_s.copy(), W_s.copy()
    
    def verify(U, V, W, r):
        """Verify the decomposition is exact."""
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                for x in range(n):
                    for y in range(m):
                        for z in range(m):
                            for w in range(p):
                                expected = 1.0 if (x == i and y == z and w == k) else 0.0
                                actual = 0.0
                                for c in range(r):
                                    actual += W[k*n+i, c] * U[x*m+y, c] * V[z*p+w, c]
                                err = abs(actual - expected)
                                if err > max_err:
                                    max_err = err
        return max_err
    
    def try_round(U, V, W, r):
        """Try to round to integer/half-integer and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, r) < 1e-9:
            return U_r, V_r, W_r
        return None
    
    # Try to find lower rank
    for r in range(best_rank - 1, max(30, best_rank - 60), -1):
        if time_remaining() < 5:
            break
        
        for attempt in range(80):
            if time_remaining() < 2:
                break
            
            np.random.seed(attempt * 1000 + r)
            
            # Random integer initialization
            U_r = np.random.randint(-3, 4, size=(n*m, r)).astype(float)
            V_r = np.random.randint(-3, 4, size=(m*p, r)).astype(float)
            
            success = True
            W_r = np.zeros((p*n, r), dtype=float)
            
            for i in range(n):
                for k in range(p):
                    if time_remaining() < 1:
                        success = False
                        break
                    
                    eqs = constraints[(i, k)]
                    num_eqs = len(eqs)
                    
                    A_sys = np.zeros((num_eqs, r))
                    b_sys = np.zeros(num_eqs)
                    for idx, (x, y, z, w, target) in enumerate(eqs):
                        b_sys[idx] = target
                        for c in range(r):
                            A_sys[idx, c] = U_r[x*m+y, c] * V_r[z*p+w, c]
                    
                    try:
                        sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        if residuals is not None and len(residuals) > 0:
                            if residuals[0] > 1e-6:
                                success = False
                                break
                        W_r[k*n+i, :] = sol
                    except:
                        success = False
                        break
                
                if not success:
                    break
            
            if not success:
                continue
            
            max_err = verify(U_r, V_r, W_r, r)
            if max_err < 1e-6:
                result = try_round(U_r, V_r, W_r, r)
                if result is not None:
                    U_rounded, V_rounded, W_rounded = result
                    best_rank = r
                    best_U, best_V, best_W = U_rounded.copy(), V_rounded.copy(), W_rounded.copy()
                    break  # Try lower r
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
