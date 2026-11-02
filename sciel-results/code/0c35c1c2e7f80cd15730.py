"""Matrix multiplication tensor decomposition for 4x5 * 5x5."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school

    # Try to find a lower rank decomposition
    # We iterate r downwards
    # We can't go lower than 1, but practically, rank is at least max(n,m,p) maybe?
    # Let's try down to 20 to be safe and fast
    
    for r in range(best_rank - 1, 19, -1):
        if time.time() > deadline - 5:
            break
            
        # Initialize with first r columns of schoolbook
        U = U_school[:, :r].copy()
        V = V_school[:, :r].copy()
        W = W_school[:, :r].copy()
        
        # ALS Optimization
        # We perform a few iterations
        # Note: We need to be careful with singular matrices in lstsq
        try:
            for _iter in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Solve for U
                # For each i, j':
                # System: sum_r U[i*m+j', r] * (V[j''*p+k, r] * W[k*n+i, r]) = delta[j', j'']
                # Unknowns: U[i*m+j', :] (size r)
                # Equations: for each j'' in 0..m-1, k in 0..p-1
                for i in range(n):
                    for j_prime in range(m):
                        # Build A (mp x r) and b (mp x 1)
                        A = np.zeros((m * p, r))
                        b = np.zeros(m * p)
                        idx = 0
                        for j_double in range(m):
                            for k in range(p):
                                # Coefficient for U[i*m+j_prime, r] is V[j_double*p+k, r] * W[k*n+i, r]
                                A[idx, :] = V[j_double * p + k, :] * W[k * n + i, :]
                                b[idx] = 1.0 if j_double == j_prime else 0.0
                                idx += 1
                        # Solve A U_col = b
                        # Use lstsq for stability
                        sol, residuals, rank_val, sv = np.linalg.lstsq(A, b, rcond=None)
                        U[i * m + j_prime, :] = sol
                        
                # Solve for V
                # For each j_double, k:
                # System: sum_r V[j_double*p+k, r] * (U[i*m+j_prime, r] * W[k*n+i, r]) = delta[j', j'']
                # Unknowns: V[j_double*p+k, :] (size r)
                # Equations: for each i in 0..n-1, j_prime in 0..m-1
                for j_double in range(m):
                    for k in range(p):
                        A = np.zeros((n * m, r))
                        b = np.zeros(n * m)
                        idx = 0
                        for i in range(n):
                            for j_prime in range(m):
                                A[idx, :] = U[i * m + j_prime, :] * W[k * n + i, :]
                                b[idx] = 1.0 if j_double == j_prime else 0.0
                                idx += 1
                        sol, residuals, rank_val, sv = np.linalg.lstsq(A, b, rcond=None)
                        V[j_double * p + k, :] = sol

                # Solve for W
                # For each k, i:
                # System: sum_r W[k*n+i, r] * (U[i*m+j_prime, r] * V[j_double*p+k, r]) = delta[j', j'']
                # Unknowns: W[k*n+i, :] (size r)
                # Equations: for each j_prime in 0..m-1, j_double in 0..m-1
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m * m, r))
                        b = np.zeros(m * m)
                        idx = 0
                        for j_prime in range(m):
                            for j_double in range(m):
                                A[idx, :] = U[i * m + j_prime, :] * V[j_double * p + k, :]
                                b[idx] = 1.0 if j_prime == j_double else 0.0
                                idx += 1
                        sol, residuals, rank_val, sv = np.linalg.lstsq(A, b, rcond=None)
                        W[k * n + i, :] = sol

            # Check if solution is close to integer/half-integer
            # Snap to nearest half-integer
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            # Verify exactness
            # sum_r U[i*m+j', r] V[j''*p+k, r] W[k*n+i, r] == delta[j', j'']
            valid = True
            for i in range(n):
                for j_prime in range(m):
                    for j_double in range(m):
                        for k in range(p):
                            s = np.dot(U_snap[i * m + j_prime, :], V_snap[j_double * p + k, :] * W_snap[k * n + i, :])
                            target = 1.0 if j_prime == j_double else 0.0
                            if abs(s - target) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
                
            if valid:
                best_rank = r
                best_result = {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
                # We can break if we find a very low rank, but let's try to go lower if time permits
                # However, lower rank is harder to find. If we found one, it's a good improvement.
                # Let's continue to see if we can go lower, but maybe with fewer iterations?
                # For now, break to save time, as finding the absolute minimum is hard.
                # Actually, let's just break. The first valid one found is likely good enough for a score improvement.
                break
                
        except Exception as e:
            # If optimization fails, continue to next rank
            pass

    return best_result
# EVOLVE-BLOCK-END
