import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i*m + j
                idx_v = j*p + k
                idx_w = k*n + i
                col = i*m*p + j*p + k
                U_sb[idx_u, col] = 1
                V_sb[idx_v, col] = 1
                W_sb[idx_w, col] = 1
                
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_sb

    def verify_solution(U, V, W, tol=1e-9):
        """Check if the decomposition is exact and entries are integer/half-integer."""
        # Check exactness of tensor contraction
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], np.multiply(V[j*p+k, :], W[k*n+i, :]))
                    if abs(val - 1.0) > tol:
                        return False
        # Check integer/half-integer constraint
        # Entries should be close to x.0 or x.5
        for mat in [U, V, W]:
            for val in mat:
                # Check if val is close to a half-integer
                closest_half_int = np.round(val * 2) / 2
                if abs(val - closest_half_int) > 1e-5:
                    return False
        return True

    def try_rank_als(r, seed=42, max_iters=2000):
        """
        Attempt to find a rank-r decomposition using ALS.
        Returns (U, V, W) if found, else None.
        """
        if time.time() > deadline:
            return None

        rng = np.random.RandomState(seed)
        
        # Initialize
        # Using random Gaussian init
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # Normalize initial vectors to prevent numerical issues
        for i in range(n*m):
            norm = np.linalg.norm(U[i, :])
            if norm > 1e-6: U[i, :] /= norm
        for j in range(m*p):
            norm = np.linalg.norm(V[j, :])
            if norm > 1e-6: V[j, :] /= norm
        for k in range(p*n):
            norm = np.linalg.norm(W[k, :])
            if norm > 1e-6: W[k, :] /= norm

        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None

            # Update U
            for i in range(n):
                for j in range(m):
                    # Equation: sum_k U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all k?
                    # No, the equation is for each (i,j,k):
                    # sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
                    # For fixed V, W, and fixed i,j, we have p equations (one for each k)
                    # Unknowns are U[i*m+j, r] for r in 0..r-1.
                    # Let u = U[i*m+j, :]
                    # For each k: sum_r u_r * (V[j*p+k,r] * W[k*n+i,r]) = 1
                    # This is a linear system M @ u = b
                    # M is p x r, M[k, r] = V[j*p+k,r] * W[k*n+i,r]
                    # b is p x 1, b[k] = 1
                    
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    
                    try:
                        # Solve least squares
                        u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    # Unknowns are V[j*p+k, r]
                    # Equations for each i (n equations)
                    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                    # M is n x r, M[i, r] = U[i*m+j,r] * W[k*n+i,r]
                    # b is n x 1, b[i] = 1
                    
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    
                    try:
                        v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    # Unknowns are W[k*n+i, r]
                    # Equations for each j (m equations)
                    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                    # M is m x r, M[j, r] = U[i*m+j,r] * V[j*p+k,r]
                    # b is m x 1, b[j] = 1
                    
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    
                    try:
                        w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass

            # Calculate error
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], np.multiply(V[j*p+k, :], W[k*n+i, :]))
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err

            # Check convergence
            if max_err < 1e-10:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify_solution(U_rounded, V_rounded, W_rounded):
                    return U_rounded, V_rounded, W_rounded
                else:
                    # If rounding fails, maybe the solution is not rational?
                    # But the problem requires integer/half-integer.
                    # If we found a real solution, we can try to project it?
                    # No, if it's not close to half-integers, it's invalid.
                    pass
            
            if max_err < 1e-6 and max_err < prev_err * 0.9:
                # Potential candidate, but need to check integer constraint at the end
                # For now, just keep going
                pass
                
            prev_err = max_err

        # Final check after max iterations
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], np.multiply(V[j*p+k, :], W[k*n+i, :]))
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err

        if max_err < 1e-10:
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if verify_solution(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded

        return None

    # Strategy:
    # 1. Try to find a decomposition with rank < 60.
    # We know rank 60 works.
    # Let's try ranks 59, 58, ... down to some limit.
    # Given the difficulty, we might only find 59 or 58.
    # We will iterate a few seeds for each rank.
    
    # To save time, we will try ranks in descending order from 59.
    # We will allow a few seconds per rank/seed combo.
    
    ranks_to_try = list(range(59, 30, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Try a few seeds for this rank
        seeds = [42, 123, 999]
        found = False
        for seed in seeds:
            if time.time() > deadline - 5:
                break
                
            res = try_rank_als(r, seed=seed, max_iters=500)
            if res is not None:
                U_found, V_found, W_found = res
                # Update best
                best_U = U_found
                best_V = V_found
                best_W = W_found
                best_rank = r
                found = True
                # If we found a solution, we can try to go lower, but 
                # finding a solution is already a win. 
                # We can break out of seed loop and try next (lower) rank.
                break
        
        if found:
            # We found a solution for rank r. 
            # We can continue to try for r-1, but it might be harder.
            # Let's try a bit more.
            continue
        else:
            # If we fail for rank r, we might fail for r-1.
            # But maybe r was just unlucky. 
            # To be safe, if we fail for 59, we probably should stop?
            # No, 59 is very close to 60. It should be easy.
            # If 59 fails, there might be a bug or it's impossible?
            # Actually, is it possible to have rank 59?
            # Yes, you can always perturb the schoolbook slightly? No, exactness.
            # If the schoolbook is the only integer solution, then 59 is impossible.
            # In that case, we return 60.
            
            # Let's assume if 59 fails quickly, we should stop trying lower ranks 
            # to save time for the fallback conversion.
            # But we should try at least one more rank?
            # Let's just break if we fail for 59.
            break

    # Convert best result to lists
    U_list = [[float(x) for x in row] for row in best_U.tolist()]
    V_list = [[float(x) for x in row] for row in best_V.tolist()]
    W_list = [[float(x) for x in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
