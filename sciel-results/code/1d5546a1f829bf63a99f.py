import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    max_time = 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    # 2x3x4 matrix multiplication
    # Target tensor T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p)
    T = np.ones((n, m, p))
    
    def verify_decomposition(U, V, W, R):
        """Verify if U, V, W form a valid decomposition of T."""
        try:
            U3 = np.array(U).reshape(n, m, R)
            V3 = np.array(V).reshape(m, p, R)
            W3 = np.array(W).reshape(p, n, R)
            # A_approx[i, j, k] = sum_c U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
            A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            return np.allclose(A_approx, T, atol=1e-9)
        except Exception:
            return False

    def try_rank(R, time_budget):
        """Try to find a rank-R decomposition using least squares optimization."""
        end_time = start_time + time_budget
        
        def unpack(params):
            U = params[:R * n * m].reshape(R, n * m).T
            V = params[R * n * m : R * (n * m + m * p)].reshape(R, m * p).T
            W = params[R * (n * m + m * p) :].reshape(R, p * n).T
            return U, V, W

        def residuals(params):
            U, V, W = unpack(params)
            U3 = U.reshape(n, m, R)
            V3 = V.reshape(m, p, R)
            W3 = W.reshape(p, n, R)
            A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            return (T - A_approx).flatten()

        num_params = R * (n * m + m * p + p * n)
        
        # Number of restarts depends on time budget
        # Estimate time per restart: ~0.1s to 1s
        max_restarts = int(time_budget / 0.5)
        max_restarts = max(10, min(max_restarts, 100))
        
        for attempt in range(max_restarts):
            if time.time() > end_time:
                break
            
            # Initialize with random values from the allowed set
            # Using a mix of integers and half-integers
            init_params = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=num_params)
            
            try:
                from scipy.optimize import least_squares
                # Use 'trf' method which is robust
                sol = least_squares(residuals, init_params, method='trf', max_nfev=100)
                
                if sol.success or sol.cost < 1e-8:
                    # Check if the solution is close to a valid integer/half-integer decomposition
                    # Try rounding to nearest 0.5
                    rounded_params = np.round(sol.x * 2) / 2.0
                    res_check = residuals(rounded_params)
                    if np.linalg.norm(res_check) < 1e-6:
                        U, V, W = unpack(rounded_params)
                        if verify_decomposition(U, V, W, R):
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": R}
                    
                    # Also check if the raw solution is exact enough (in case rounding failed slightly)
                    # But the problem requires integer/half-integer entries, so we must round.
                    # If raw is exact but not integer/half-integer, it's invalid per contract.
                    
            except Exception:
                pass
                
        return None

    # Search for lower ranks
    # Known optimal rank for 2x3x4 is 20.
    # Try 20 with a large budget.
    if time.time() - start_time < 100:
        res = try_rank(20, 95)
        if res:
            return res

    # Try 21
    if time.time() - start_time < 120:
        res = try_rank(21, 15)
        if res:
            return res

    # Try 22
    if time.time() - start_time < 130:
        res = try_rank(22, 5)
        if res:
            return res

    # Try 23
    if time.time() - start_time < 140:
        res = try_rank(23, 5)
        if res:
            return res

    # Fallback: Schoolbook decomposition (rank 24)
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
