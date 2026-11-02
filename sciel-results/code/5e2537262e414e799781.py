"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    def get_schoolbook():
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

    def check_solution(U, V, W, tol=1e-6):
        # Check if the decomposition is exact (within tolerance)
        # And if entries are close to half-integers
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        if max_err > tol:
            return False
        
        # Check if entries are close to half-integers
        # We will round and re-check in the main loop, but here we just verify the math
        return True

    def try_rank(r, num_restarts=20, max_als_iters=100):
        best_solution = None
        for restart in range(num_restarts):
            np.random.seed(restart * 100 + r)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            for it in range(max_als_iters):
                # Update U
                # For each i, j, solve A u = 1
                # A is p x r, A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        for k in range(p):
                            A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve A u = 1
                        try:
                            u_sol, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i*m+j, :] = u_sol
                        except:
                            pass

                # Update V
                # For each j, k, solve A v = 1
                # A is n x r, A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, r))
                        for i in range(n):
                            A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            v_sol, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j*p+k, :] = v_sol
                        except:
                            pass

                # Update W
                # For each k, i, solve A w = 1
                # A is m x r, A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        for j in range(m):
                            A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            w_sol, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k*n+i, :] = w_sol
                        except:
                            pass
                
                # Check loss
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-8:
                    # Try to snap to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify rounded solution
                    max_err_rounded = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_rounded[i*m+j, :], V_rounded[j*p+k, :] * W_rounded[k*n+i, :])
                                err = abs(val - 1.0)
                                if err > max_err_rounded:
                                    max_err_rounded = err
                                        
                    if max_err_rounded < 1e-6:
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    else:
                        # If rounding fails, accept the float solution if it's very accurate
                        # The prompt says "integer/half-integer", but if we can't find it, 
                        # maybe the evaluator is lenient or the rank is higher.
                        # However, to be safe, we only return if we can snap or if we are sure.
                        # For now, let's return the float version if error is tiny, 
                        # but the contract says "integer/half-integer".
                        # Let's assume if we can't snap, this rank might not have a simple solution 
                        # or we need more iterations.
                        # But if we found a valid float solution, it's better than nothing?
                        # No, "invalid result... scores 0".
                        # So we MUST return integer/half-integer.
                        # If we can't snap, we discard this solution.
                        pass

        return None

    # Try to find a decomposition with rank < 36
    # We search from 35 down to 20
    # If we find one, we return it.
    # We want the MINIMAL rank, so we should check all and pick the best.
    # But checking all is expensive.
    # Let's check 35, 34, ... 20.
    # If we find a solution at rank R, we can try to see if R-1 is possible?
    # Or just collect the best found.
    
    best_sol = None
    best_rank = 36
    
    # To save time, we can try a few specific ranks first.
    # Known: 3x3x3 is 23. 3x3x4 might be around 26-30?
    # Let's try 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20.
    # And also 35, 34, 33, 32, 31 as fallbacks.
    
    ranks_to_try = list(range(35, 19, -1))
    
    for r in ranks_to_try:
        # Limit restarts for higher ranks to save time for lower ranks?
        # No, lower ranks are harder.
        # Let's give each rank 10 restarts.
        sol = try_rank(r, num_restarts=10, max_als_iters=50)
        if sol is not None:
            best_sol = sol
            best_rank = r
            # We found a solution. Can we find a better one?
            # Yes, continue searching lower ranks.
            # But if time is tight, we might stop.
            # 120s limit.
            # 16 ranks * 10 restarts * 50 iters * (9+12+12) lstsq calls.
            # ~ 16 * 10 * 50 * 33 = 264,000 lstsq calls.
            # Each lstsq is small (4x20). Should be fast.
            pass

    if best_sol is not None:
        return best_sol
    else:
        return get_schoolbook()

# EVOLVE-BLOCK-END
