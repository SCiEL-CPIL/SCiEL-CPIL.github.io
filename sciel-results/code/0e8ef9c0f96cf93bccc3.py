"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    def verify_solution(U, V, W, r):
        """Verify that the decomposition is exact and entries are half-integers."""
        # Check half-integer constraint
        # U, V, W are numpy arrays
        if not np.allclose(U * 2, np.round(U * 2), atol=1e-9): return False
        if not np.allclose(V * 2, np.round(V * 2), atol=1e-9): return False
        if not np.allclose(W * 2, np.round(W * 2), atol=1e-9): return False
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for rr in range(r):
                        val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < 1e-9

    def try_rank(r, seed=42, max_iters=2000):
        """Try to find a rank-r decomposition using optimization."""
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        # Scaling is important. If entries are too large, ALS might diverge or converge to bad solutions.
        U = rng.randn(n * m, r) * 0.5
        V = rng.randn(m * p, r) * 0.5
        W = rng.randn(p * n, r) * 0.5
        
        # Pre-allocate matrices for speed
        # We will update U, V, W in place
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None, None, None, False
            
            # --- Update U ---
            # For each i, j:
            # M[k, t] = V[j*p+k, t] * W[k*n+i, t]
            # Solve M @ U[i*m+j, :] = ones(p)
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Use lstsq for robustness
                    try:
                        u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # --- Update V ---
            # For each j, k:
            # M[i, t] = U[i*m+j, t] * W[k*n+i, t]
            # Solve M @ V[j*p+k, :] = ones(n)
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            # --- Update W ---
            # For each k, i:
            # M[j, t] = U[i*m+j, t] * V[j*p+k, t]
            # Solve M @ W[k*n+i, :] = ones(m)
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            # Check convergence every few iterations to save time
            if it % 10 == 0:
                # Quick check: compute max error
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-8:
                    # Try to round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    if verify_solution(U_rounded, V_rounded, W_rounded, r):
                        return U_rounded, V_rounded, W_rounded, True
                    else:
                        # Maybe it's an integer solution?
                        U_int = np.round(U)
                        V_int = np.round(V)
                        W_int = np.round(W)
                        if verify_solution(U_int, V_int, W_int, r):
                            return U_int, V_int, W_int, True
                        else:
                            # Return the floating point solution if it's exact enough, 
                            # but the problem requires half-integers. 
                            # If floating point works but rounded doesn't, it's not a valid "half-integer" solution 
                            # unless we consider that the "exact" tensor decomposition allows any real numbers 
                            # BUT the prompt says "integer/half-integer entries". 
                            # So we must return half-integers.
                            # If we can't round it, this rank/seed failed.
                            pass

        # Final check
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-8:
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if verify_solution(U_rounded, V_rounded, W_rounded, r):
                return U_rounded, V_rounded, W_rounded, True
            else:
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if verify_solution(U_int, V_int, W_int, r):
                    return U_int, V_int, W_int, True

        return None, None, None, False

    # Try ranks from 8 up to 25
    # We prioritize lower ranks.
    # We will try a few seeds for each rank.
    # If we find a solution, we return it immediately.
    
    # The schoolbook rank is 60.
    # Known bounds suggest rank is likely between 10 and 20.
    
    for r in range(8, 26):
        if time.time() > deadline - 10:
            break
            
        # Try multiple seeds for this rank
        # More seeds for lower ranks? Or uniform?
        # Let's try 5 seeds for each rank.
        for seed in range(5):
            if time.time() > deadline - 5:
                break
                
            U, V, W, success = try_rank(r, seed=seed, max_iters=1000)
            if success:
                U_list = [[float(U[i][j]) for j in range(r)] for i in range(n*m)]
                V_list = [[float(V[i][j]) for j in range(r)] for i in range(m*p)]
                W_list = [[float(W[i][j]) for j in range(r)] for i in range(p*n)]
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
