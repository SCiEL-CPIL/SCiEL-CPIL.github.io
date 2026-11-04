import numpy as np
import time
import sys
from scipy.optimize import least_squares

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
        """Try to find a rank-R decomposition using least squares optimization and discrete refinement."""
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
        
        # Discrete set of allowed values
        discrete_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        # Estimate max restarts based on time budget
        # Each restart involves optimization which takes ~0.1-0.5s
        max_restarts = int(time_budget / 0.2)
        max_restarts = max(50, min(max_restarts, 200))
        
        for attempt in range(max_restarts):
            if time.time() > end_time:
                break
            
            # Initialize with random discrete values
            init_params = np.random.choice(discrete_vals, size=num_params)
            
            try:
                # Use least squares to refine the continuous parameters
                sol = least_squares(residuals, init_params, method='trf', max_nfev=200)
                
                # Check if the solution is close to a valid integer/half-integer decomposition
                # Try rounding to nearest 0.5
                rounded_params = np.round(sol.x * 2) / 2.0
                
                # Verify the rounded solution
                if verify_decomposition(rounded_params[:R*n*m].reshape(R, n*m).T, 
                                        rounded_params[R*n*m:R*(n*m+m*p)].reshape(R, m*p).T, 
                                        rounded_params[R*(n*m+m*p):].reshape(R, p*n).T, R):
                    U, V, W = unpack(rounded_params)
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": R}
                
                # If rounding didn't work exactly, try a local discrete search around the rounded solution
                # This is a slow step, so we only do it if the continuous solution was very good
                if sol.cost < 1e-6:
                    # Perform a limited discrete local search
                    # We perturb each parameter by +/- 0.5 and check if the error decreases
                    # This is too slow for all params, so we do a few random perturbations
                    for _ in range(10):
                        perturbed = rounded_params.copy()
                        # Pick a few indices to perturb
                        indices = np.random.choice(num_params, size=min(5, num_params), replace=False)
                        for idx in indices:
                            # Try adding/subtracting 0.5
                            orig_val = perturbed[idx]
                            for delta in [-0.5, 0.5]:
                                new_val = orig_val + delta
                                if new_val in discrete_vals:
                                    perturbed[idx] = new_val
                                    if verify_decomposition(perturbed[:R*n*m].reshape(R, n*m).T, 
                                                            perturbed[R*n*m:R*(n*m+m*p)].reshape(R, m*p).T, 
                                                            perturbed[R*(n*m+m*p):].reshape(R, p*n).T, R):
                                        U, V, W = unpack(perturbed)
                                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist(), "rank": R}
                                    perturbed[idx] = orig_val # revert

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
