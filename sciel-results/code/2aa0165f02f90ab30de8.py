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
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    start_time = time.time()
    time_limit = 110  # seconds

    def is_half_integer(matrix, tol=1e-6):
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

    # Search for lower ranks, starting from highest (closest to schoolbook) down to lower.
    # This is because lower ranks are harder to find.
    # If we find R=59, we record it. Then we try R=58.
    # If we find R=58, we record it.
    # We return the best (lowest) rank found.
    
    # We iterate R from schoolbook-1 down to 40.
    for R in range(rank_school - 1, 39, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Determine number of restarts based on remaining time
        # Estimate: 1 restart ~ 0.01s. 100 restarts ~ 1s.
        # We have ~110s.
        # Let's do 200 restarts per rank if time permits.
        num_restarts = 200
        
        for r_idx in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Random U, V with small integers
            # Using {-1, 0, 1}
            U_rand = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(nm)]
            V_rand = [[random.choice([-1, 0, 1]) for _ in range(R)] for _ in range(mp)]
            
            A = np.zeros((n * m * p, R * n * p))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U_rand[i * m + j]
                        v_row = V_rand[j * p + k]
                        for c in range(R):
                            coeff = u_row[c] * v_row[c]
                            if coeff != 0:
                                w_idx = (k * n + i) * R + c
                                A[eq_idx, w_idx] = coeff
                        eq_idx += 1
            
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            if residuals is not None and len(residuals) > 0 and residuals[0] > 1e-5:
                continue
                
            W_sol = W_flat.reshape((p * n, R))
            
            ok, W_nice = is_half_integer(W_sol)
            if ok:
                if verify_decomposition(U_rand, V_rand, W_nice, R):
                    best_rank = R
                    best_U = U_rand
                    best_V = V_rand
                    best_W = W_nice
                    # Do not break, keep searching for lower R
                    # But if we find a very low rank, we might want to stop?
                    # No, we want the minimum.
                    pass

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
