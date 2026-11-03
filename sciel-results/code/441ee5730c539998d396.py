"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time
import itertools

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
    
    def verify(U, V, W, r, tol=1e-8):
        """Verify that the decomposition is exact"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
                    if abs(s - 1.0) > tol:
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
                # Round to nearest half-integer first, then to fraction
                x_rounded = round(x * 2) / 2
                row_frac.append(Fraction(x_rounded))
            result.append(row_frac)
        return result
    
    def try_random_search(r, num_trials=30, max_iters=80):
        """Try random search for rank-r decomposition using ALS"""
        t0 = time.time()
        for trial in range(num_trials):
            if time.time() > deadline - 15.0:
                return None
            
            np.random.seed(trial * 12345 + 67890 + r)
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
            
            if max_err < 1e-6:
                # Try to find exact solution by rounding to half-integers
                # Try multiple rounding strategies
                for denom in [2, 4, 8, 16]:
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
                
                # Also try half-integer rounding
                U_h = np.round(U * 2) / 2
                V_h = np.round(V * 2) / 2
                W_h = np.round(W * 2) / 2
                
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r_idx in range(r):
                                val += U_h[i * m + j, r_idx] * V_h[j * p + k, r_idx] * W_h[k * n + i, r_idx]
                            if abs(val - 1.0) > 1e-10:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    res_U = to_fraction_list(U_h)
                    res_V = to_fraction_list(V_h)
                    res_W = to_fraction_list(W_h)
                    if res_U is not None and res_V is not None and res_W is not None:
                        return {
                            "U": res_U,
                            "V": res_V,
                            "W": res_W
                        }
        
        return None
    
    def try_gradient_descent(r, num_restarts=5, max_iters=1500):
        """Try gradient descent for rank-r decomposition"""
        rng = np.random.RandomState(42 + r)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15.0:
                return None
            
            U = rng.randn(n * m, r) * 0.5
            V = rng.randn(m * p, r) * 0.5
            W = rng.randn(p * n, r) * 0.5
            
            lr = 0.01
            prev_loss = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline - 15.0:
                    return None
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u = U[i * m + j]
                            v = V[j * p + k]
                            w = W[k * n + i]
                            val = np.dot(u * w, v)
                            err = val - 1.0
                            dU[i * m + j] += err * (w * v)
                            dV[j * p + k] += err * (u * w)
                            dW[k * n + i] += err * (u * v)
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Compute loss
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j] * W[k * n + i], V[j * p + k])
                            loss += (val - 1.0) ** 2
                
                # Adaptive learning rate
                if loss < prev_loss:
                    lr = min(lr * 1.05, 0.1)
                else:
                    lr *= 0.7
                    U += lr * dU
                    V += lr * dV
                    W += lr * dW
                    loss = prev_loss
                prev_loss = loss
                
                if loss < 1e-10:
                    break
            
            # Try rounding to half-integers
            if loss < 1e-4:
                U_h = np.round(U * 2) / 2
                V_h = np.round(V * 2) / 2
                W_h = np.round(W * 2) / 2
                
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_h[i * m + j] * W_h[k * n + i], V_h[j * p + k])
                            if abs(val - 1.0) > 1e-10:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                
                if ok:
                    res_U = to_fraction_list(U_h)
                    res_V = to_fraction_list(V_h)
                    res_W = to_fraction_list(W_h)
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
        
        # Try ALS first
        result = try_random_search(r, num_trials=20, max_iters=60)
        if result is not None:
            best_rank = r
            best_result = result
            break
        
        # Try gradient descent
        result = try_gradient_descent(r, num_restarts=3, max_iters=1000)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    return best_result
# EVOLVE-BLOCK-END
