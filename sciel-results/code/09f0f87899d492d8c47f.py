"""Matrix multiplication 3x3x5 rank-minimal decomposition."""
DIMENSIONS = [3, 3, 5]

def construct():
    import time
    import numpy as np

    deadline = time.time() + 130
    n, m, p = DIMENSIONS

    # Schoolbook rank: n*m*p = 45
    # Build the standard rank-45 decomposition
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

    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank

    # Try to find a lower-rank decomposition using randomized search
    # The constraint: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for all i in [0,n), j in [0,m), k in [0,p)
    #
    # Strategy: For a given rank R, randomly initialize V and W with small
    # integer/half-integer values, then solve for U via least squares.
    # Check if the result is exact and if U has acceptable entries.
    
    rng = np.random.default_rng(12345)
    
    # Try ranks from 44 down to 15 (theoretical lower bound is max(nm, mp, np) = 15)
    for R in range(rank - 1, 14, -1):
        if time.time() > deadline - 10:
            break
        
        # Number of attempts for this rank
        max_attempts = 200 if R >= 40 else 500
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Randomly initialize V and W with entries from {-1, -0.5, 0, 0.5, 1}
            # This keeps things simple and in the allowed entry set
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            V_rand = rng.choice(vals, size=(m * p, R))
            W_rand = rng.choice(vals, size=(p * n, R))
            
            # For each (i, j), the equation is:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # This gives p equations per (i,j) pair, with R unknowns (U[i*m+j, :])
            # Total: n*m*p = 45 equations, n*m*R = 9R unknowns
            
            # Build the linear system: for each (i,j,k),
            # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
            
            # Let's build a matrix A of size (n*m*p, R) where each row corresponds
            # to a specific (i,j,k) and the row is V[j*p+k, :] * W[k*n+i, :]
            # But U depends on (i,j), not k. So we need to group by (i,j).
            
            # For a fixed (i,j), we have p equations:
            # For k=0..p-1: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is: M_ij @ U[i*m+j, :] = ones(p)
            # where M_ij[k, r] = V[j*p+k, r] * W[k*n+i, r]
            
            # So for each (i,j), we solve M_ij @ u_ij = ones(p)
            # where u_ij = U[i*m+j, :]
            
            # If M_ij is invertible (p x p submatrix) or has full column rank, we can solve.
            # Since R >= p (we need R >= 5 for p=5), the system is underdetermined.
            # We can find a solution if ones(p) is in the column space of M_ij.
            
            # Let's try to solve for each (i,j) independently using lstsq.
            
            U_sol = np.zeros((n * m, R))
            valid = True
            
            for i in range(n):
                for j in range(m):
                    # Build M_ij of shape (p, R)
                    M_ij = np.zeros((p, R))
                    for k in range(p):
                        M_ij[k, :] = V_rand[j * p + k, :] * W_rand[k * n + i, :]
                    
                    # Solve M_ij @ u = ones(p)
                    # Use lstsq for underdetermined system
                    try:
                        u_ij, residuals, rank_m, sv = np.linalg.lstsq(M_ij, np.ones(p), rcond=None)
                        # Check residual
                        res = np.sum((M_ij @ u_ij - np.ones(p))**2)
                        if res > 1e-6:
                            valid = False
                            break
                        U_sol[i * m + j, :] = u_ij
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify the full decomposition
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U_sol[i * m + j, r] * V_rand[j * p + k, r] * W_rand[k * n + i, r]
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-8:
                # Check if all entries are integer or half-integer (within tolerance)
                def is_half_integer(arr, tol=1e-6):
                    scaled = arr * 2
                    rounded = np.round(scaled)
                    return np.all(np.abs(scaled - rounded) < tol)
                
                if is_half_integer(U_sol) and is_half_integer(V_rand) and is_half_integer(W_rand):
                    # Snap to exact half-integers
                    U_sol = np.round(U_sol * 2) / 2
                    V_rand = np.round(V_rand * 2) / 2
                    W_rand = np.round(W_rand * 2) / 2
                    
                    # Final verification
                    max_err2 = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for r in range(R):
                                    val += U_sol[i * m + j, r] * V_rand[j * p + k, r] * W_rand[k * n + i, r]
                                max_err2 = max(max_err2, abs(val - 1.0))
                    
                    if max_err2 < 1e-10:
                        best_result = {"U": U_sol.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
                        best_rank = R
                        # Found a valid decomposition at this rank, return it
                        return best_result
            
            # If we found a real solution but not half-integer, try rounding U
            # and re-verifying
            U_rounded = np.round(U_sol * 2) / 2
            max_err3 = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U_rounded[i * m + j, r] * V_rand[j * p + k, r] * W_rand[k * n + i, r]
                        max_err3 = max(max_err3, abs(val - 1.0))
            
            if max_err3 < 1e-10 and is_half_integer(V_rand) and is_half_integer(W_rand):
                best_result = {"U": U_rounded.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
                best_rank = R
                return best_result
    
    return best_result
