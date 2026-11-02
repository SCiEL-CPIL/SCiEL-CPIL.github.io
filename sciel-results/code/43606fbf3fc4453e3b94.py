import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    start = time.time()
    DEADLINE = 135  # safety margin before 150s limit

    n, m, p = 3, 3, 6

    def verify(U, V, W, r):
        """
        Verify that the decomposition exactly computes matrix multiplication.
        U: (n*m, r), V: (m*p, r), W: (p*n, r)
        For random A (n x m) and B (m x p):
        C[i, k] = sum_j A[i, j] * B[j, k]
        Decomposition: C[i, k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        """
        rng = np.random.default_rng(42)
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for _ in range(5):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            C_true = A @ B
            
            C_comp = np.zeros((n, p), dtype=float)
            for i in range(n):
                for k in range(p):
                    val = 0.0
                    for j in range(m):
                        u_row = U_arr[i * m + j, :]
                        v_row = V_arr[j * p + k, :]
                        w_val = W_arr[k * n + i, :]
                        val += np.sum(u_row * v_row * w_val)
                    C_comp[i, k] = val
            
            if not np.allclose(C_comp, C_true, atol=1e-6):
                return False
        return True

    # Build the guaranteed schoolbook decomposition (Rank 54)
    r_school = n * m * p
    U_school = [[0.0] * r_school for _ in range(n * m)]
    V_school = [[0.0] * r_school for _ in range(m * p)]
    W_school = [[0.0] * r_school for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][c] = 1.0
                V_school[j * p + k][c] = 1.0
                W_school[k * n + i][c] = 1.0
                c += 1

    # The tensor T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p)
    # We want to find U, V, W of rank r such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k
    
    def residuals(params, r):
        """
        params: flattened array of length r*(n*m + m*p + p*n)
        Returns residual vector of length n*m*p
        """
        nm = n * m
        mp = m * p
        pn = p * n
        
        U = params[:r*nm].reshape(nm, r)
        V = params[r*nm:r*nm+r*mp].reshape(mp, r)
        W = params[r*nm+r*mp:].reshape(pn, r)
        
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    res[i, j, k] = val - 1.0
        return res.flatten()

    def jacobian(params, r):
        """
        Numerical Jacobian. Might be slow but r is small.
        Actually, let's try analytical or just let scipy handle it with finite differences if needed.
        Given the small size, finite difference is likely fine but might be slow.
        Let's try to implement a simple forward diff if scipy is too slow, or just use least_squares default.
        """
        # For now, let scipy use finite differences. 
        # To speed up, we can provide a sparse structure or analytical jacobian.
        # But let's first see if least_squares converges.
        pass

    best_rank = r_school
    best_U, best_V, best_W = U_school, V_school, W_school

    # Try to find lower ranks
    # Lower bound is roughly max(nm, mp, pn) = 18.
    # Known lower bound for 3x3x3 is 19. For 3x3x6, it's likely higher.
    # Let's try ranks from 53 down to 40.
    
    # We will iterate over candidate ranks.
    for r_try in range(r_school - 1, 39, -1):
        if time.time() - start > DEADLINE:
            break
            
        found = False
        # Try a few random restarts for each rank
        rng = np.random.default_rng(42 + r_try)
        
        for restart in range(3):
            if time.time() - start > DEADLINE:
                break
                
            # Initialize parameters
            # Scale is important. 
            # If we initialize with 1/sqrt(r), the sum of r terms might be O(1).
            scale = 1.0 / np.sqrt(r_try)
            params0 = rng.standard_normal(r_try * (n*m + m*p + p*n)) * scale
            
            try:
                # Use least_squares to minimize sum of squared residuals
                # We want residuals to be 0.
                res = least_squares(
                    residuals, 
                    params0, 
                    args=(r_try,), 
                    method='trf', 
                    max_nfev=5000, 
                    ftol=1e-12, 
                    xtol=1e-12, 
                    gtol=1e-12
                )
                
                if res.success or res.cost < 1e-10:
                    # Extract U, V, W
                    nm = n * m
                    mp = m * p
                    pn = p * n
                    U = res.x[:r_try*nm].reshape(nm, r_try)
                    V = res.x[r_try*nm:r_try*nm+r_try*mp].reshape(mp, r_try)
                    W = res.x[r_try*nm+r_try*mp:].reshape(pn, r_try)
                    
                    # Convert to lists of lists for return format
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    
                    # Verify
                    if verify(U_list, V_list, W_list, r_try):
                        best_rank = r_try
                        best_U, best_V, best_W = U_list, V_list, W_list
                        found = True
                        break # Move to next lower rank
                        
            except Exception:
                pass

        if found:
            continue # Try even lower rank
        else:
            # If we fail at r_try, it's possible lower ranks also fail, 
            # but not necessarily. However, if we fail at 53, 52, etc., 
            # we might just stop to save time if we've tried several.
            # Given the difficulty, let's just break if we fail at the first few.
            # Actually, let's just continue but limit total time.
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
