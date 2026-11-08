"""
Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses structured local search: start from rank-68 Strassen+Direct,
attempt rank reduction via column elimination with coefficient re-solving.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 150s kill
    
    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)
        
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)
        
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])
    
    def direct_term(i, j, k, U_t, V_t, W_t):
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])
    
    U_t = []
    V_t = []
    W_t = []
    
    # 4 Strassen blocks for A[0:2,0:2] * B[0:2,k:k+2]
    for k_off in [0, 2, 4, 6]:
        strassen_decomp(0, 0, 0, k_off, U_t, V_t, W_t)
    
    # Direct terms for remaining parts
    for k_off in [0, 2, 4, 6]:
        for i in range(2):
            for k in range(k_off, k_off + 2):
                direct_term(i, 2, k, U_t, V_t, W_t)
        for j in range(2):
            for k in range(k_off, k_off + 2):
                direct_term(2, j, k, U_t, V_t, W_t)
        for k in range(k_off, k_off + 2):
            direct_term(2, 2, k, U_t, V_t, W_t)
    
    base_rank = len(W_t)
    
    # Build base factor matrices
    U_base = np.zeros((n * m, base_rank))
    V_base = np.zeros((m * p, base_rank))
    W_base = np.zeros((p * n, base_rank))
    
    for idx in range(base_rank):
        for (i, j), coeff in U_t[idx]:
            U_base[i * m + j, idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V_base[j * p + k, idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W_base[k * n + i, idx] = coeff
    
    def verify_factors(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_factors(U_base, V_base, W_base):
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = base_rank
    
    # Strategy: Try to reduce rank by eliminating one column at a time
    # For each candidate column to eliminate, solve for new coefficients
    # in the remaining columns that compensate for the removed term.
    
    # The tensor equation: sum_r U[:,r] V[:,r] W[:,r] = T
    # If we remove column r0, we need: sum_{r != r0} U[:,r] V[:,r] W[:,r] = T - U[:,r0] V[:,r0] W[:,r0]
    # This is generally not solvable with the same number of terms minus one,
    # but we can try: keep all other columns fixed and see if the residual
    # can be absorbed by adjusting one other column.
    
    # Alternative: Try pairwise column merging.
    # If columns r1 and r2 have proportional U, V, or W patterns,
    # they might be mergeable.
    
    # Better structural approach: Try different Strassen block arrangements.
    # For 3x3 x 3x8, we can try:
    # - Use Strassen on A[0:2,0:2] x B[0:2,0:8] (all 8 cols at once? No, Strassen is 2x2)
    # - Try using a rank-8 method for 2x2 x 2x8? Not standard.
    
    # Key insight: The 4 Strassen blocks use the SAME A[0:2,0:2] but different B[0:2,k:k+2].
    # Can we share any of the 7 Strassen terms across blocks?
    # No, because the B inputs are different.
    
    # Try: Can we replace some direct terms with shared Strassen-like terms?
    # For example, A[0:2,2] * B[2,0:8] is 2x1 x 1x8 = 16 direct terms.
    # Could we do better? A[0:2,2] is a 2x1 matrix, B[2,0:8] is 1x8.
    # This is just a vector-matrix product: C[i,k] = A[i,2]*B[2,k] for i=0,1; k=0..7.
    # That's 16 terms, and each is independent. Can't reduce below 16 for this subproblem.
    
    # Similarly A[2,0:2] * B[0:2,0:8]: 1x2 x 2x8 = 16 terms.
    # A[2,2] * B[2,0:8]: 1x1 x 1x8 = 8 terms.
    
    # Total direct: 16 + 16 + 8 = 40. Strassen: 4*7 = 28. Total: 68.
    
    # The only way to reduce is to find a better decomposition of the full problem.
    # Known lower bound for 3x3x3 is 19 (Bini et al. conjecture, not proven).
    # For 3x3x8, lower bound is at least max(3*3, 3*8, 3*8) = 24, but more tightly
    # the rank is at least the dimension of the space, which is complex.
    
    # Practical approach: Try random perturbations and see if we can find rank 67.
    # Use a smarter method: for each pair of columns, check if they can be merged.
    
    # For a merge to work, we need:
    # U[:,r1] V[:,r1] W[:,r1] + U[:,r2] V[:,r2] W[:,r2] = U[:,r_new] V[:,r_new] W[:,r_new]
    # This requires the sum of two rank-1 tensors to be rank-1, which is very restrictive.
    
    # Instead, try: remove one column, and adjust ALL remaining columns slightly.
    # This is a nonlinear optimization. Use scipy if available, otherwise gradient descent.
    
    try:
        from scipy.optimize import least_squares
        has_scipy = True
    except ImportError:
        has_scipy = False
    
    if has_scipy and time.time() < DEADLINE - 10:
        # Try to find rank-67 decomposition via optimization
        target_rank = base_rank - 1
        
        # Precompute the tensor structure
        # T[I,J,K] = 1 where I=i*m+j, J=j*p+k, K=k*n+i
        tensor_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_indices.append((i*m+j, j*p+k, k*n+i))
        
        # For rank R, parameters are U (nm x R), V (mp x R), W (pn x R)
        # Total params: (nm + mp + pn) * R
        
        # Objective: minimize sum over all (I,J,K) of (sum_r U[I,r]V[J,r]W[K,r] - 1)^2
        
        def objective(params):
            U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
            V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_flat = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
            
            residuals = []
            for I, J, K in tensor_indices:
                s = np.sum(U_flat[I] * V_flat[J] * W_flat[K])
                residuals.append(s - 1.0)
            return np.array(residuals)
        
        # Initialize from base, drop one column
        best_result = None
        for drop_idx in range(min(base_rank, 20)):  # Try first 20 columns to drop
            if time.time() > DEADLINE - 5:
                break
            
            keep_mask = np.ones(base_rank, dtype=bool)
            keep_mask[drop_idx] = False
            
            U_init = U_base[:, keep_mask].copy()
            V_init = V_base[:, keep_mask].copy()
            W_init = W_base[:, keep_mask].copy()
            
            # Add small noise
            U_init += np.random.normal(0, 0.05, U_init.shape)
            V_init += np.random.normal(0, 0.05, V_init.shape)
            W_init += np.random.normal(0, 0.05, W_init.shape)
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(objective, params_init, method='lm', max_nfev=1000)
                if result.success or np.max(np.abs(result.fun)) < 1e-6:
                    U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    # Round to half-integers
                    U_round = np.round(U_opt * 2) / 2
                    V_round = np.round(V_opt * 2) / 2
                    W_round = np.round(W_opt * 2) / 2
                    
                    if verify_factors(U_round, V_round, W_round):
                        best_U, best_V, best_W = U_round, V_round, W_round
                        best_rank = target_rank
                        break
                    elif verify_factors(U_opt, V_opt, W_opt):
                        # Accept even if not nice, but prefer nice
                        if best_result is None:
                            best_result = (U_opt, V_opt, W_opt)
            except Exception:
                pass
        
        if best_rank == base_rank and best_result is not None:
            best_U, best_V, best_W = best_result
    
    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
