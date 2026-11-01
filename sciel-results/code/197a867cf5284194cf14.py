"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

import time

# EVOLVE-BLOCK-START

import numpy as np
from scipy.optimize import minimize

def _verify_decomposition(U, V, W, n=2, m=2, p=6, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor: T[i*m+j, j'*p+k, k'*n+i'] should equal delta_{j,j'} * delta_{k,k'}
    # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Where a = i*m+j, b = j'*p+k, c = k'*n+i'
    
    # The multiplication tensor has T[i,j,k] = 1 iff the indices are consistent
    # For our indexing: T[(i,j),(j',k),(k',i')] = delta_{j,j'} * delta_{k,k'}
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            # T index: a = i*m+j, b = jp*p+k, c = kp*n+ip
                            a = i * m + j
                            b = jp * p + k
                            c = kp * n + ip
                            
                            # Expected value
                            expected = 1.0 if (j == jp and k == kp) else 0.0
                            
                            # Computed value
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            
                            err = abs(val - expected)
                            if err > max_err:
                                max_err = err
    
    return max_err < tol, max_err

def _try_rank(rank, n=2, m=2, p=6, max_iters=20000, tol=1e-9, time_budget=100.0):
    """Try to find a rank-`rank` decomposition using optimization."""
    start_time = time.time()
    
    # Total number of variables: rank*(n*m + m*p + p*n)
    num_vars = rank * (n * m + m * p + p * n)
    
    # Initialize with the Strassen-based solution if rank >= 21, else random
    if rank >= 21:
        # Use Strassen as initial guess
        U_init = [[0] * rank for _ in range(n * m)]
        V_init = [[0] * rank for _ in range(m * p)]
        W_init = [[0] * rank for _ in range(p * n)]
        
        for b in range(3):
            k0 = 2 * b
            k1 = 2 * b + 1
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            w_c00 = k0 * n + 0
            w_c10 = k0 * n + 1
            w_c01 = k1 * n + 0
            w_c11 = k1 * n + 1
            
            # Step 0
            r = b * 7 + 0
            U_init[0][r] = 1; U_init[3][r] = 1
            V_init[v_e][r] = 1; V_init[v_h][r] = 1
            W_init[w_c00][r] = 1; W_init[w_c11][r] = 1
            # Step 1
            r = b * 7 + 1
            U_init[2][r] = 1; U_init[3][r] = 1
            V_init[v_e][r] = 1
            W_init[w_c10][r] = 1; W_init[w_c11][r] = -1
            # Step 2
            r = b * 7 + 2
            U_init[0][r] = 1
            V_init[v_f][r] = 1; V_init[v_h][r] = -1
            W_init[w_c01][r] = 1; W_init[w_c11][r] = 1
            # Step 3
            r = b * 7 + 3
            U_init[3][r] = 1
            V_init[v_g][r] = 1; V_init[v_e][r] = -1
            W_init[w_c00][r] = 1; W_init[w_c10][r] = 1
            # Step 4
            r = b * 7 + 4
            U_init[0][r] = 1; U_init[1][r] = 1
            V_init[v_h][r] = 1
            W_init[w_c00][r] = -1; W_init[w_c01][r] = 1
            # Step 5
            r = b * 7 + 5
            U_init[2][r] = 1; U_init[0][r] = -1
            V_init[v_e][r] = 1; V_init[v_f][r] = 1
            W_init[w_c11][r] = 1
            # Step 6
            r = b * 7 + 6
            U_init[1][r] = 1; U_init[3][r] = -1
            V_init[v_g][r] = 1; V_init[v_h][r] = 1
            W_init[w_c00][r] = 1
        
        x0 = np.concatenate([
            np.array(U_init).flatten(),
            np.array(V_init).flatten(),
            np.array(W_init).flatten()
        ])
    else:
        # Random initialization
        np.random.seed(42)
        x0 = np.random.randn(num_vars) * 0.5
    
    # Objective: sum of squared errors over all tensor entries
    def objective(x):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            return 1e10  # Penalize timeout
        
        U_flat = x[:rank * n * m]
        V_flat = x[rank * n * m : rank * n * m + rank * m * p]
        W_flat = x[rank * n * m + rank * m * p :]
        
        U = U_flat.reshape(n * m, rank)
        V = V_flat.reshape(m * p, rank)
        W = W_flat.reshape(p * n, rank)
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                a = i * m + j
                                b = jp * p + k
                                c = kp * n + ip
                                expected = 1.0 if (j == jp and k == kp) else 0.0
                                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                                err += (val - expected) ** 2
        
        return err
    
    # Try L-BFGS-B
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': max_iters, 'maxfun': max_iters, 'ftol': 1e-14})
        
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            return None
        
        final_err = result.fun
        if final_err < tol * tol:  # Since we're summing squares
            # Extract solution
            U_sol = result.x[:rank * n * m].reshape(n * m, rank)
            V_sol = result.x[rank * n * m : rank * n * m + rank * m * p].reshape(m * p, rank)
            W_sol = result.x[rank * n * m + rank * m * p :].reshape(p * n, rank)
            
            # Round to nearest half-integer or integer if close
            U_round = np.round(U_sol * 2) / 2
            V_round = np.round(V_sol * 2) / 2
            W_round = np.round(W_sol * 2) / 2
            
            # Verify rounded solution
            valid, max_err = _verify_decomposition(
                U_round.tolist(), V_round.tolist(), W_round.tolist(), 
                n=n, m=m, p=p, tol=1e-9
            )
            if valid:
                return U_round.tolist(), V_round.tolist(), W_round.tolist()
            
            # Try integer rounding
            U_round2 = np.round(U_sol)
            V_round2 = np.round(V_sol)
            W_round2 = np.round(W_sol)
            valid2, max_err2 = _verify_decomposition(
                U_round2.tolist(), V_round2.tolist(), W_round2.tolist(),
                n=n, m=m, p=p, tol=1e-9
            )
            if valid2:
                return U_round2.tolist(), V_round2.tolist(), W_round2.tolist()
            
            # Return as-is if close enough
            if final_err < 1e-6:
                return U_sol.tolist(), V_sol.tolist(), W_sol.tolist()
        
        return None
    except Exception:
        return None

def _strassen_solution(n=2, m=2, p=6):
    """Return the known rank-21 Strassen-based solution."""
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(3):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        w_c00 = k0 * n + 0
        w_c10 = k0 * n + 1
        w_c01 = k1 * n + 0
        w_c11 = k1 * n + 1
        
        r = b * 7 + 0
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        
        r = b * 7 + 1
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        
        r = b * 7 + 2
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        
        r = b * 7 + 3
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        
        r = b * 7 + 4
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        
        r = b * 7 + 5
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        
        r = b * 7 + 6
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to find lower rank solutions
    best_solution = None
    best_rank = 21
    
    # Get the known rank-21 solution as fallback
    U21, V21, W21 = _strassen_solution(n, m, p)
    best_solution = (U21, V21, W21)
    
    # Try rank 20
    if time.time() - start_time < 120:
        result = _try_rank(20, n, m, p, max_iters=50000, time_budget=50.0)
        if result is not None:
            best_solution = result
            best_rank = 20
    
    # Try rank 19
    if time.time() - start_time < 120 and best_rank == 21:
        result = _try_rank(19, n, m, p, max_iters=50000, time_budget=50.0)
        if result is not None:
            best_solution = result
            best_rank = 19
    
    U, V, W = best_solution
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
