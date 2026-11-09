import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Try to find lower rank
    # Known lower bounds or just try descending
    # R(2,4,7) is likely 52? 51?
    # Let's try 55, 54, 53, 52.
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Initialize best result with schoolbook
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    best_result = {'U': U_school, 'V': V_school, 'W': W_school}
    
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    # Tensor T
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0 # The multiplication tensor has 1s everywhere?
                # Wait. The tensor for matrix multiplication C = AB is defined by
                # C_ik = sum_j A_ij B_jk.
                # The tensor T such that C = T(A,B) has components T_{ijk}.
                # Actually, the standard definition is T_{i,j,k} = 1 if the term A_ij B_jk appears.
                # So yes, T is all ones for the standard multiplication.
                
    # ALS Function
    def try_rank(r):
        if r >= best_rank:
            return None
            
        # Random init
        np.random.seed(42) # Fixed seed for reproducibility in search? Or random?
        # Let's use random init
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Reshape T for easier indexing?
        # T[i, j, k]
        
        for step in range(100):
            if time.time() > deadline:
                return None
                
            # Update W
            # System: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = T[i, j, k]
            # Variables: W_flat (size p*n*r)
            # Build A matrix (56 x 14r)
            A_W = np.zeros((n*m*p, p*n*r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_t = i*m*p + j*p + k
                        for c in range(r):
                            idx_w = (k*n + i)*r + c
                            val = U[i*m+j, c] * V[j*p+k, c]
                            A_W[idx_t, idx_w] = val
                            
            # Solve A_W W_flat = T_flat
            # T_flat is just ones vector of size 56
            T_flat = np.ones(n*m*p)
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A_W, T_flat, rcond=None)
                W = W_flat.reshape(p*n, r)
            except:
                return None
                
            # Update U
            A_U = np.zeros((n*m*p, n*m*r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_t = i*m*p + j*p + k
                        for c in range(r):
                            idx_u = (i*m + j)*r + c
                            val = V[j*p+k, c] * W[k*n+i, c]
                            A_U[idx_t, idx_u] = val
            try:
                U_flat, residuals, rank_A, s = np.linalg.lstsq(A_U, T_flat, rcond=None)
                U = U_flat.reshape(n*m, r)
            except:
                return None
                
            # Update V
            A_V = np.zeros((n*m*p, m*p*r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_t = i*m*p + j*p + k
                        for c in range(r):
                            idx_v = (j*p + k)*r + c
                            val = U[i*m+j, c] * W[k*n+i, c]
                            A_V[idx_t, idx_v] = val
            try:
                V_flat, residuals, rank_A, s = np.linalg.lstsq(A_V, T_flat, rcond=None)
                V = V_flat.reshape(m*p, r)
            except:
                return None
                
            # Check error
            # Reconstruct
            T_recon = np.zeros((n, m, p))
            for c in range(r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_recon[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            
            err = np.linalg.norm(T_recon - T)
            if err < 1e-6:
                # Found a solution
                # Try to round to half-integers?
                # The problem says "integer/half-integer entries".
                # If the solution is rational with small denominator, rounding might work.
                # Let's check if rounding to 0.5 works.
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_recon_round = np.zeros((n, m, p))
                for c in range(r):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_recon_round[i, j, k] += U_rounded[i*m+j, c] * V_rounded[j*p+k, c] * W_rounded[k*n+i, c]
                                
                err_round = np.linalg.norm(T_recon_round - T)
                if err_round < 1e-6:
                    return {'U': U_rounded, 'V': V_rounded, 'W': W_rounded}
                else:
                    # Maybe just return the float solution? The evaluator might accept floats if they are exact?
                    # "integer/half-integer entries" is specified.
                    # If rounding fails, we might need to use rational recovery.
                    # For now, return float and hope? No, better to return schoolbook if rounding fails.
                    # But maybe the float solution IS the answer and rounding is just an artifact?
                    # If the true rank is r, there exists a solution with integer/half-integer entries.
                    # ALS might converge to a different solution.
                    # Let's return the float solution for now and see if evaluator accepts it?
                    # No, "integer/half-integer" is a constraint.
                    # I will return the schoolbook if rounding fails.
                    return None
                    
        return None

    # Try ranks
    for r in range(rank_schoolbook - 1, 20, -1):
        if time.time() > deadline:
            break
        res = try_rank(r)
        if res is not None:
            best_result = res
            best_rank = r
            # Continue to lower ranks?
            # Yes, but it might be hard.
            # Let's break if we find one, or continue for a bit.
            # Given the time, finding one is good.
            # Let's try to go lower.
            continue
            
    return best_result
