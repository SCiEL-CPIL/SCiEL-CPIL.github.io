import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Build the tensor T as a vector or use the equation directly
    # Equation: sum_r U[i,j,r] V[j,k,r] W[i,k,r] = 1 if (i,j,k) is valid, else 0?
    # Actually, the tensor T has T[(i,j),(j',k),(i',k')] = 1 if i=i', j=j', k=k'.
    # So for each (i,j,k), we have a constraint:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # Note: The index in W is k*n + i based on the prompt "W[k*n+i,r]".
    # The prompt says: "output factor follows the official trace convention W[k*n+i,r]".
    # And U is (n*m, rank), V is (m*p, rank).
    # The standard schoolbook uses U[i*m+j], V[j*p+k], W[k*n+i].
    
    # We want to find U, V, W of rank R < 60.
    
    def verify(U, V, W, R):
        # Check if U, V, W satisfy the equations
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Constraint for all i in [0,n), j in [0,m), k in [0,p):
        # sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # This is a set of n*m*p equations.
        # We can check the residual.
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try to find a decomposition of rank R
    # We will try R from 50 to 59.
    # Use a simple randomized alternating least squares.
    
    # Precompute indices for speed
    idx_u = [(i, j) for i in range(n) for j in range(m)]
    idx_v = [(j, k) for j in range(m) for k in range(p)]
    idx_w = [(k, i) for k in range(p) for i in range(n)]
    
    # The constraints are coupled.
    # For a fixed r, the term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    
    # Let's try a specific construction if possible, or just return schoolbook if search fails.
    # Given the difficulty and time limit, and that 60 is a safe valid answer,
    # and no inspiration provided, the probability of finding a better rank quickly is low.
    # However, if we can find rank 59, we score 1.0000 * (60/59) > 1.0.
    
    # Let's do a very limited search.
    start_time = time.time()
    deadline = 110 # seconds
    
    best_rank = rank_schoolbook
    best_result = None
    
    # Create schoolbook result
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to improve
    # We can try to merge two terms?
    # If we have two rank-1 tensors, can we combine them?
    # Not easily.
    
    # Let's try to solve for rank 59 with random restarts.
    # This is unlikely to work well with simple ALS in 100s for a non-convex problem.
    # But let's try a few iterations.
    
    def solve_rank(R, max_iters=20, restarts=5):
        for _ in range(restarts):
            if time.time() > deadline:
                return None
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            for it in range(max_iters):
                # Update U
                # For each (i,j), we have equations for all k:
                # sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                # Let A_{k,r} = V[j*p+k, r] * W[k*n+i, r].
                # Then sum_r U[i*m+j, r] * A_{k,r} = 1 for all k.
                # This is a linear system for U[i*m+j, :].
                # A is (p, R). We want U_row such that U_row @ A.T = [1]*p.
                # If R >= p, we can solve. If R < p, it's overdetermined.
                # Use least squares.
                
                for i in range(n):
                    for j in range(m):
                        # Build A for this (i,j)
                        # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        A = np.zeros((p, R))
                        for k in range(p):
                            for r in range(R):
                                A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        # Solve U[i*m+j, :] @ A.T = ones(p)
                        # Equivalent to A.T @ U_col = ones(p) ? No.
                        # Let u be the vector U[i*m+j, :].
                        # sum_r u_r * A[k, r] = 1 for all k.
                        # This is A @ u = 1? No, A is (p, R).
                        # A @ u is a vector of size p.
                        # We want A @ u = [1, 1, ..., 1].
                        try:
                            u, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i*m+j, :] = u
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, R)) # i dimension
                        for i in range(n):
                            for r in range(R):
                                A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        # A @ v = [1]*n
                        try:
                            v, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(n), rcond=None)
                            V[j*p+k, :] = v
                        except:
                            pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, R)) # j dimension
                        for j in range(m):
                            for r in range(R):
                                A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        # A @ w = [1]*m
                        try:
                            w, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            W[k*n+i, :] = w
                        except:
                            pass

            # Check residual
            # Convert to lists for verification
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if verify(U_list, V_list, W_list, R):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try ranks from 59 down to 50
    for R in range(59, 49, -1):
        if time.time() > deadline:
            break
        res = solve_rank(R, max_iters=10, restarts=2)
        if res:
            # Round to nearest integer or half-integer?
            # The problem says "integer/half-integer entries".
            # If the solution is exact, it should be close to such values.
            # Let's try to round.
            U_r = [[round(x * 2) / 2 for x in row] for row in res["U"]]
            V_r = [[round(x * 2) / 2 for x in row] for row in res["V"]]
            W_r = [[round(x * 2) / 2 for x in row] for row in res["W"]]
            if verify(U_r, V_r, W_r, R):
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                best_rank = R
                break # Found a better one, stop searching for lower ranks? 
                # Actually, we want the minimum rank. So we should try lower ranks.
                # But we are iterating from high to low. So we found a valid one.
                # We can continue to try lower ranks if time permits.
                # But 59 is a good improvement. Let's try to go lower.
                # The loop continues to R-1.
    
    return best_result
# EVOLVE-BLOCK-END
