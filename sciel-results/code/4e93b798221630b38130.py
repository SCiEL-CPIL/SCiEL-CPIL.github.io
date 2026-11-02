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
    time_limit = 115  # seconds, safety margin
    
    def is_nice_matrix(matrix, tol=1e-9):
        """Check if matrix entries are close to half-integers. Returns (bool, nice_matrix)"""
        arr = np.array(matrix, dtype=float)
        doubled = arr * 2
        rounded = np.round(doubled)
        if not np.allclose(doubled, rounded, atol=tol):
            return False, None
        nice = rounded / 2.0
        # Convert to list of lists of int/float
        result = []
        for row in nice:
            res_row = []
            for val in row:
                if val == int(val):
                    res_row.append(int(val))
                else:
                    res_row.append(val)
            result.append(res_row)
        return True, result

    def verify_decomposition(U, V, W, rank):
        # Vectorized verification
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i * m + j]
                    v_val = V_arr[j * p + k]
                    w_val = W_arr[k * n + i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Search for lower rank
    # We will try to solve for U given random V, W.
    # System: A U_flat = b
    # A shape: (60, 12*R)
    # b shape: (60,)
    
    ranks_to_try = [20, 22, 24, 26, 28, 30, 35, 40, 50]
    
    for R in ranks_to_try:
        if R >= best_rank:
            continue
            
        # Number of restarts
        num_restarts = 10000
        for r_idx in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Random V, W with entries in {-1, 0, 1}
            V_rand = np.random.choice([-1, 0, 1], size=(mp, R))
            W_rand = np.random.choice([-1, 0, 1], size=(pn, R))
            
            # Build A
            A = np.zeros((n * m * p, nm * R))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        v_row = V_rand[j * p + k]
                        w_row = W_rand[k * n + i]
                        coeffs = v_row * w_row
                        start_col = (i * m + j) * R
                        end_col = start_col + R
                        A[eq_idx, start_col:end_col] = coeffs
                        eq_idx += 1
            
            # Solve
            try:
                U_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
                
            # Check residual to avoid unnecessary nice-checks
            if len(residuals) > 0 and residuals[0] > 1e-6:
                continue
                
            U_sol = U_flat.reshape((nm, R))
            
            # Check if U_sol is nice
            ok, U_nice = is_nice_matrix(U_sol)
            if ok:
                # Verify with the nice values
                if verify_decomposition(U_nice, V_rand.tolist(), W_rand.tolist(), R):
                    best_rank = R
                    best_U = U_nice
                    best_V = V_rand.tolist()
                    best_W = W_rand.tolist()
                    return {"U": best_U, "V": best_V, "W": best_W}

        if time.time() - start_time > time_limit:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
