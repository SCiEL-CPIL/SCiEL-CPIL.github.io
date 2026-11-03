"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    start_time = time.time()
    deadline = start_time + 120.0
    
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
        """Verify that the decomposition is exact"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r_idx in range(r):
                        s += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                    if s != 1:
                        return False
        return True
    
    def to_fraction_list(matrix):
        """Convert numpy array to list of Fractions"""
        result = []
        for row in matrix:
            row_frac = []
            for x in row:
                if not np.isfinite(x):
                    return None
                row_frac.append(Fraction(x).limit_denominator(1000000))
            result.append(row_frac)
        return result
    
    def try_random_search(r, num_trials=100, max_iters=50):
        """Try random search for rank-r decomposition"""
        t0 = time.time()
        for trial in range(num_trials):
            if time.time() > deadline - 15.0:
                return None
            
            np.random.seed(trial * 12345 + 67890)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS refinement
            for it in range(max_iters):
                if time.time() > deadline - 15.0:
                    return None
                
                # Update U
                for idx in range(n * m):
                    i, j = divmod(idx, m)
                    vec = np.zeros(r)
                    for k in range(p):
                        vec += V[j * p + k] * W[k * n + i]
                    U[idx] = vec
                
                # Update V
                for idx in range(m * p):
                    j, k = divmod(idx, p)
                    vec = np.zeros(r)
                    for i in range(n):
                        vec += U[i * m + j] * W[k * n + i]
                    V[idx] = vec
                
                # Update W
                for idx in range(p * n):
                    k, i = divmod(idx, n)
                    vec = np.zeros(r)
                    for j in range(m):
                        vec += U[i * m + j] * V[j * p + k]
                    W[idx] = vec
            
            # Check error and finiteness
            if not np.all(np.isfinite(U)) or not np.all(np.isfinite(V)) or not np.all(np.isfinite(W)):
                continue

            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r_idx in range(r):
                            val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < 1e-8:
                # Try to find exact solution
                # Round to nearest rational with small denominator
                for denom in [1, 2, 4, 8, 16, 32]:
                    U_frac = np.round(U * denom) / denom
                    V_frac = np.round(V * denom) / denom
                    W_frac = np.round(W * denom) / denom
                    
                    # Verify
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
                            return {
                                "U": res_U,
                                "V": res_V,
                                "W": res_W
                            }
        
        return None
    
    # Try ranks from lower bound up
    lower_bound = max(n * m, m * p, p * n)  # 15
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 20.0:
            break
        result = try_random_search(r, num_trials=50, max_iters=30)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    return best_result
# EVOLVE-BLOCK-END
