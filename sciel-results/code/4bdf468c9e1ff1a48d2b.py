"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # Schoolbook decomposition as fallback
    rank_school = n * m * p  # 45
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def verify(U, V, W, r):
        """Verify that the decomposition is exact using Fractions"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r_idx in range(r):
                        s += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                    if s != 1:
                        return False
        return True

    def to_fraction_list(matrix):
        """Convert numpy array to list of Fractions, returning None if non-finite"""
        result = []
        for row in matrix:
            row_frac = []
            for x in row:
                if not np.isfinite(x):
                    return None
                # Use limit_denominator to get a clean fraction
                row_frac.append(Fraction(x).limit_denominator(1000000))
            result.append(row_frac)
        return result

    def compute_residual(U, V, W, r):
        """Compute max absolute residual of the decomposition"""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_idx in range(r):
                        val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                    max_err = max(max_err, abs(val - 1.0))
        return max_err

    def als_step(U, V, W, r, max_iters=100, tol=1e-10):
        """
        Alternating Least Squares refinement for tensor decomposition.
        Updates U, V, W to minimize the residual.
        Returns (U, V, W, converged)
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                return U, V, W, False
            
            old_U = U.copy()
            old_V = V.copy()
            old_W = W.copy()
            
            # Update U: for each (i,j), U[i*m+j] = argmin ||T_ij - V_j ⊙ W_i||
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    # We want U[idx] such that sum_r U[idx,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                    # This is a linear system: A * U[idx] = b
                    # where A[r1, r2] = sum_k V[j*p+k, r1] * W[k*n+i, r2]
                    # and b[r2] = ... wait, let's think more carefully.
                    
                    # sum_r U[idx, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for each k
                    # Let's define for each k: sum_r U[idx, r] * C_k[r] = 1 where C_k[r] = V[j*p+k,r]*W[k*n+i,r]
                    # So we have a system: M * U[idx] = 1_vector
                    # where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    
                    M = np.zeros((p, r))
                    for k in range(p):
                        for r_idx in range(r):
                            M[k, r_idx] = V[j * p + k, r_idx] * W[k * n + i, r_idx]
                    
                    # Solve M * U[idx] = ones(p)
                    try:
                        U[idx] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            
            # Update V: for each (j,k), V[j*p+k] = argmin ||T_jk - U_j ⊙ W_k||
            for j in range(m):
                for k in range(p):
                    idx = j * p + k
                    # sum_r V[idx, r] * U[i*m+j, r] * W[k*n+i, r] = 1 for each i
                    # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    M = np.zeros((n, r))
                    for i in range(n):
                        for r_idx in range(r):
                            M[i, r_idx] = U[i * m + j, r_idx] * W[k * n + i, r_idx]
                    
                    try:
                        V[idx] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            
            # Update W: for each (k,i), W[k*n+i] = argmin ||T_ki - U_i ⊙ V_k||
            for k in range(p):
                for i in range(n):
                    idx = k * n + i
                    # sum_r W[idx, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for each j
                    # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    M = np.zeros((m, r))
                    for j in range(m):
                        for r_idx in range(r):
                            M[j, r_idx] = U[i * m + j, r_idx] * V[j * p + k, r_idx]
                    
                    try:
                        W[idx] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            
            # Check convergence
            if np.allclose(U, old_U, atol=tol) and np.allclose(V, old_V, atol=tol) and np.allclose(W, old_W, atol=tol):
                return U, V, W, True
        
        return U, V, W, False

    def try_random_search(r, num_trials=20, max_als_iters=150):
        """Try random search for rank-r decomposition with ALS refinement"""
        t0 = time.time()
        for trial in range(num_trials):
            if time.time() > deadline - 15.0:
                return None
            
            np.random.seed(trial * 12345 + 67890 + r * 7)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS refinement
            U, V, W, converged = als_step(U, V, W, r, max_iters=max_als_iters)
            
            # Check error and finiteness
            if not np.all(np.isfinite(U)) or not np.all(np.isfinite(V)) or not np.all(np.isfinite(W)):
                continue

            max_err = compute_residual(U, V, W, r)
            
            if max_err < 1e-8:
                # Try to find exact solution by rounding to rationals
                found = False
                for denom in [1, 2, 4, 8, 16, 32, 64]:
                    U_frac = np.round(U * denom) / denom
                    V_frac = np.round(V * denom) / denom
                    W_frac = np.round(W * denom) / denom
                    
                    # Verify with high precision
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for r_idx in range(r):
                                    val += U_frac[i * m + j, r_idx] * V_frac[j * p + k, r_idx] * W_frac[k * n + i, r_idx]
                                if abs(val - 1.0) > 1e-10:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    
                    if ok:
                        res_U = to_fraction_list(U_frac)
                        res_V = to_fraction_list(V_frac)
                        res_W = to_fraction_list(W_frac)
                        if res_U is not None and res_V is not None and res_W is not None:
                            # Final exact verification
                            if verify(res_U, res_V, res_W, r):
                                return {
                                    "U": res_U,
                                    "V": res_V,
                                    "W": res_W
                                }
        
        return None

    # Try ranks from lower bound up
    # Theoretical lower bound: max(n*m, m*p, p*n) = 15
    # But for 3x3x5, known bounds suggest higher. Let's try from 15 upward.
    lower_bound = max(n * m, m * p, p * n)  # 15
    
    # We can also try a slightly higher start if we have time
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 25.0:
            break
        
        # Reduce trials as r increases (harder to find)
        if r <= 20:
            num_trials = 30
        elif r <= 30:
            num_trials = 20
        else:
            num_trials = 10
            
        max_als_iters = 200 if r <= 20 else 100
        
        result = try_random_search(r, num_trials=num_trials, max_als_iters=max_als_iters)
        if result is not None:
            best_rank = r
            best_result = result
            break  # Found the minimal rank, we're done
    
    return best_result
# EVOLVE-BLOCK-END
