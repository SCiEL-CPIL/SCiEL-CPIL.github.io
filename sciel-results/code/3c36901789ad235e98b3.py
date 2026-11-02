import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
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
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    start_time = time.time()
    deadline = start_time + 120  # 120s safety margin

    # Try to find a lower rank decomposition
    # Ranks to try
    ranks_to_try = [100, 95, 90, 85, 80, 75, 70]
    
    # Precompute indices for equations
    # Equations: sum_r W[k*n+i, r] * U[i*m+j1, r] * V[j2*p+k, r] = delta[j1, j2]
    # Total equations: n * p * m * m = 3 * 7 * 5 * 5 = 525
    
    def check_solution(U, V, W, R):
        # Check if the decomposition is exact
        # Residual for each i, k, j1, j2
        max_err = 0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = 0
                        for r in range(R):
                            s += W[k * n + i, r] * U[i * m + j1, r] * V[j2 * p + k, r]
                        target = 1.0 if j1 == j2 else 0.0
                        err = abs(s - target)
                        if err > max_err:
                            max_err = err
        return max_err

    def round_to_half(x):
        return np.round(x * 2) / 2.0

    def als_decompose(R, max_iters=20, tol=1e-8):
        # Initialize U, V, W randomly
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None, None, None, 1.0
            
            # Update W
            # For each i, k, solve X * w = y
            new_W = np.zeros_like(W)
            for i in range(n):
                for k in range(p):
                    # Build X matrix of size (m*m) x R
                    X = np.zeros((m * m, R))
                    for j1 in range(m):
                        for j2 in range(m):
                            # X[j1*m + j2, r] = U[i*m+j1, r] * V[j2*p+k, r]
                            X[j1 * m + j2, :] = U[i * m + j1, :] * V[j2 * p + k, :]
                    y = np.zeros(m * m)
                    for j in range(m):
                        y[j * m + j] = 1.0
                    # Solve X w = y
                    try:
                        w, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
                        new_W[k * n + i, :] = w
                    except:
                        return None, None, None, 1.0
            W = new_W
            
            # Update U
            new_U = np.zeros_like(U)
            for i in range(n):
                for j1 in range(m):
                    # For fixed i, j1, we want to find U[i*m+j1, :]
                    # Equations involve W[k*n+i, :] and V[j2*p+k, :]
                    # Sum_k, j2, r W[k*n+i, r] U[i*m+j1, r] V[j2*p+k, r] = delta[j1, j2]
                    # This is not a simple linear system for U[i*m+j1, :] because of the sum over k and j2.
                    # Actually, for fixed i, j1, the variable is u = U[i*m+j1, :].
                    # The equations are:
                    # For each k, j2: sum_r W[k*n+i, r] u_r V[j2*p+k, r] = delta[j1, j2]
                    # This is a linear system in u.
                    # Number of equations: p * m = 7 * 5 = 35.
                    # Unknowns: R.
                    X = np.zeros((p * m, R))
                    y = np.zeros(p * m)
                    for k in range(p):
                        for j2 in range(m):
                            # Row index: k*m + j2
                            X[k * m + j2, :] = W[k * n + i, :] * V[j2 * p + k, :]
                            y[k * m + j2] = 1.0 if j2 == j1 else 0.0
                    try:
                        u, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
                        new_U[i * m + j1, :] = u
                    except:
                        return None, None, None, 1.0
            U = new_U
            
            # Update V
            new_V = np.zeros_like(V)
            for k in range(p):
                for j2 in range(m):
                    # Variable v = V[j2*p+k, :]
                    # Equations: For each i, j1: sum_r W[k*n+i, r] U[i*m+j1, r] v_r = delta[j1, j2]
                    # Number of equations: n * m = 3 * 5 = 15.
                    X = np.zeros((n * m, R))
                    y = np.zeros(n * m)
                    for i in range(n):
                        for j1 in range(m):
                            X[i * m + j1, :] = W[k * n + i, :] * U[i * m + j1, :]
                            y[i * m + j1] = 1.0 if j1 == j2 else 0.0
                    try:
                        v, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
                        new_V[j2 * p + k, :] = v
                    except:
                        return None, None, None, 1.0
            V = new_V
            
            # Check error
            err = check_solution(U, V, W, R)
            if err < tol:
                return U, V, W, err
        
        return U, V, W, check_solution(U, V, W, R)

    best_result = None
    best_rank = n * m * p
    best_U, best_V, best_W = schoolbook()["U"], schoolbook()["V"], schoolbook()["W"]
    # Convert to numpy for internal use
    best_U = np.array(best_U)
    best_V = np.array(best_V)
    best_W = np.array(best_W)

    for R in ranks_to_try:
        if time.time() > deadline:
            break
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline:
                break
            U, V, W, err = als_decompose(R)
            if U is None:
                continue
            if err < 1e-6:
                # Round to half-integers
                U_r = round_to_half(U)
                V_r = round_to_half(V)
                W_r = round_to_half(W)
                # Check rounded solution
                err_r = check_solution(U_r, V_r, W_r, R)
                if err_r < 1e-6:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                # If rounding fails, keep the float solution? 
                # Task says "integer/half-integer entries".
                # If we can't find an exact integer/half-integer one, we might fail.
                # But schoolbook is exact.
                # We should only return if it's exact.
            
            # If we found a low error but not exact, we might try to polish?
            # For now, we just track the best found.
            if err < 1e-6 and R < best_rank:
                best_rank = R
                best_U, best_V, best_W = U, V, W
    
    # If we didn't find a better one, return schoolbook
    if best_rank == n * m * p:
        return schoolbook()
    else:
        # Return the best found, hoping it's exact or close enough?
        # The evaluator checks exactness.
        # So we must ensure it's exact.
        # If the float solution is exact, we can return it? 
        # "integer/half-integer entries" is a constraint.
        # So we must return integers/half-integers.
        # If we can't round it exactly, we can't use it.
        return schoolbook()
