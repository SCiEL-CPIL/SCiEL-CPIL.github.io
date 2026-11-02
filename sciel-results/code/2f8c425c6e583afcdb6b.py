import numpy as np
import random
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    start_time = time.time()
    time_limit = 110  # seconds
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Coefficient set for U and V
    # Using {-1, 0, 1} is a common starting point for "integer" decompositions
    coeffs = [-1, 0, 1]
    
    def is_half_integer(matrix, tol=1e-9):
        for row in matrix:
            for val in row:
                nearest = round(val * 2) / 2
                if abs(val - nearest) > tol:
                    return False, None
        result = []
        for row in matrix:
            res_row = []
            for val in row:
                nearest = round(val * 2) / 2
                if nearest == int(nearest):
                    res_row.append(int(nearest))
                else:
                    res_row.append(nearest)
            result.append(res_row)
        return True, result

    def verify_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Helper to solve for W given U and V
    def solve_for_W(U, V, R):
        # A has shape (n*m*p, R*n*p)
        A = np.zeros((n * m * p, R * n * p))
        b = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(R):
                        coeff = U[i * m + j][c] * V[j * p + k][c]
                        w_idx = (k * n + i) * R + c
                        A[eq_idx, w_idx] = coeff
                    eq_idx += 1
        
        try:
            W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            # Check residual
            if residuals and residuals[0] > 1e-8:
                return None
            W_sol = W_flat.reshape((p * n, R))
            return W_sol
        except:
            return None

    # Search ranks from low to high
    # Lower bound for 3x4x5 is not trivially known, but definitely < 60.
    # Let's try 25 to 55.
    for R in range(25, 56):
        if R >= best_rank:
            continue
            
        # Number of restarts
        num_restarts = 500
        
        for r_idx in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U and V sparsely
            U_rand = [[random.choice(coeffs) for _ in range(R)] for _ in range(nm)]
            V_rand = [[random.choice(coeffs) for _ in range(R)] for _ in range(mp)]
            
            W_sol = solve_for_W(U_rand, V_rand, R)
            if W_sol is None:
                continue
                
            ok, W_nice = is_half_integer(W_sol)
            if ok:
                if verify_decomposition(U_rand, V_rand, W_nice, R):
                    best_rank = R
                    best_U = U_rand
                    best_V = V_rand
                    best_W = W_nice
                    # Found a valid one for this rank. Since we iterate R increasing, 
                    # this is the minimal rank found so far. 
                    # We can return immediately if we want the first minimal found, 
                    # but maybe a lower R is possible? No, we are iterating up.
                    # So if we find R=30, we return. If we don't find 30, we try 31.
                    return {"U": best_U, "V": best_V, "W": best_W}

        # If we didn't find it in random restarts, try local search?
        # For simplicity and speed, we stick to random restarts with more iterations.
        # If time is running out, break.
        if time.time() - start_time > time_limit:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
