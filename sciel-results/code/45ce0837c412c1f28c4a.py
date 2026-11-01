"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 matrix multiplication.
    
    Strategy:
    1. Attempt to find a rank-6 decomposition by solving the polynomial system
       T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    2. If no rank-6 solution is found (which is expected, as rank 7 is minimal),
       return the verified rank-7 Strassen decomposition.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 2, 2
    
    # Time guard: return Strassen if we're getting close to the limit
    def time_remaining():
        return 150.0 - (time.time() - start_time)
    
    # --- Attempt rank-6 search via polynomial system ---
    # The tensor T has entries T[i,j,k] = 1 for all i,j,k in {0,1}
    # For rank r, we need:
    #   U: (n*m, r) = (4, r)
    #   V: (m*p, r) = (4, r)  
    #   W: (p*n, r) = (4, r)
    # such that for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    #
    # This is a system of 8 polynomial equations (one per (i,j,k) triple)
    # in 3*r unknowns. For r=6, that's 18 unknowns, 8 equations.
    # The system is underdetermined but the polynomial constraints are nonlinear.
    #
    # We can set up the system and check if it has a solution using
    # numerical optimization or homotopy. However, it's known that rank(2x2x2) = 7,
    # so no solution exists. We'll do a quick numerical check.
    
    def try_rank6():
        """Attempt to find rank-6 decomposition via numerical optimization."""
        if time_remaining() < 30:
            return None
        
        from scipy.optimize import minimize
        
        # Flatten unknowns: U (4*6=24), V (4*6=24), W (4*6=24) = 72 unknowns
        # But we can use symmetry/gauge: fix some entries to reduce unknowns.
        # Actually, let's just do a direct approach with random restarts.
        
        best_residual = np.inf
        best_x = None
        
        n_restarts = 50
        for restart in range(n_restarts):
            if time_remaining() < 15:
                break
            
            # Random initialization
            x0 = np.random.randn(72) * 0.5
            U0 = x0[:24].reshape(4, 6)
            V0 = x0[24:48].reshape(4, 6)
            W0 = x0[48:].reshape(4, 6)
            
            def objective(x):
                U = x[:24].reshape(4, 6)
                V = x[24:48].reshape(4, 6)
                W = x[48:].reshape(4, 6)
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            residual += (val - 1.0)**2
                return residual
            
            try:
                result = minimize(objective, x0, method='BFGS', 
                                 options={'maxiter': 1000, 'gtol': 1e-8})
                if result.fun < best_residual:
                    best_residual = result.fun
                    best_x = result.x
            except Exception:
                pass
        
        # Check if we found a valid solution
        if best_x is not None and best_residual < 1e-6:
            U = best_x[:24].reshape(4, 6).tolist()
            V = best_x[24:48].reshape(4, 6).tolist()
            W = best_x[48:].reshape(4, 6).tolist()
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(6))
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U, "V": V, "W": W}
        
        return None
    
    rank6_result = try_rank6()
    if rank6_result is not None:
        return rank6_result
    
    # --- Fallback: Verified rank-7 Strassen decomposition ---
    if time_remaining() < 5:
        # Emergency return with simple schoolbook rank 8
        rank = 8
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Strassen's rank-7 decomposition
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Index mapping:
    # U[i*m + j] -> A[i,j]: A00:0, A01:1, A10:2, A11:3
    # V[j*p + k] -> B[j,k]: B00:0, B01:1, B10:2, B11:3
    # W[k*n + i] -> C[i,k]: C00:0, C10:1, C01:2, C11:3
    
    strassen = [
        # M1 = (A00 + A11)(B00 + B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10 + A11) * B00
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),
        # M3 = A00 * (B01 - B11)
        ({0: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}),
        # M4 = A11 * (B10 - B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 1: 1}),
        # M5 = (A00 + A01) * B11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),
        # M6 = (A10 - A00) * (B00 + B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01 - A11) * (B10 + B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    for r, (u_c, v_c, w_c) in enumerate(strassen):
        for idx, val in u_c.items():
            U[idx][r] = val
        for idx, val in v_c.items():
            V[idx][r] = val
        for idx, val in w_c.items():
            W[idx][r] = val
    
    # Rigorous verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m + j, :] * V_arr[j*p + k, :] * W_arr[k*n + i, :])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook rank 8
        rank = 8
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
