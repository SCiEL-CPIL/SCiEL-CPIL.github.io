import numpy as np
import time
import random
from itertools import product
import scipy.optimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target rank for 3x3x2 is 16 (total 32) or 17 (total 34).
    # Schoolbook 3x3x4 is 36.
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = n * m * p # 36
    
    # Initialize with schoolbook for 3x3x4
    U_school = np.zeros((n * m, best_rank))
    V_school = np.zeros((m * p, best_rank))
    W_school = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school

    def verify_full(U_mat, V_mat, W_mat):
        n, m, p = 3, 3, 4
        rank = U_mat.shape[1]
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(s - 1.0) > 1e-8:
                        return False
        # Check zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if (i == i2 and j == j2 and k == k2):
                        continue
                    s = 0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s) > 1e-8:
                        return False
        return True

    def search_332(rank_332):
        n_sub, m_sub, p_sub = 3, 3, 2
        n_vars = 9 * rank_332 + 6 * rank_332 + 6 * rank_332
        
        # Precompute target tensor for 3x3x2
        T_target = np.zeros((9, 6, 6))
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    a = i * 3 + j
                    b = j * 2 + k
                    c = k * 3 + i
                    T_target[a, b, c] = 1.0

        def error(x):
            U = x[:9*rank_332].reshape(9, rank_332)
            V = x[9*rank_332:15*rank_332].reshape(6, rank_332)
            W = x[15*rank_332:21*rank_332].reshape(6, rank_332)
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_approx - T_target
            return np.sum(diff**2)

        # Try multiple random restarts
        for restart in range(10):
            if time.time() > deadline - 20:
                break
            
            # Initialize with small random values
            x0 = np.random.randn(n_vars) * 0.5
            
            # Optimize
            try:
                result = scipy.optimize.minimize(error, x0, method='L-BFGS-B', 
                                                 options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-6})
            except Exception:
                continue
                
            if result.fun < 1e-8:
                U = result.x[:9*rank_332].reshape(9, rank_332)
                V = result.x[9*rank_332:15*rank_332].reshape(6, rank_332)
                W = result.x[15*rank_332:21*rank_332].reshape(6, rank_332)
                
                # Round to nearest 0.5
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                # Verify rounded version
                T_approx_r = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
                diff_r = T_approx_r - T_target
                if np.sum(diff_r**2) < 1e-8:
                    return U_round, V_round, W_round
        
        return None

    # Try ranks for 3x3x2
    # Rank 16 is the target for 32 total.
    for r in [16, 17]:
        if time.time() > deadline - 30:
            break
            
        print(f"Trying rank {r} for 3x3x2...")
        result_332 = search_332(r)
        
        if result_332 is not None:
            U_sub, V_sub, W_sub = result_332
            rank_full = 2 * r
            
            U_full = np.zeros((9, rank_full))
            V_full = np.zeros((12, rank_full))
            W_full = np.zeros((12, rank_full))
            
            U_full[:, :r] = U_sub
            V_full[:6, :r] = V_sub
            W_full[:6, :r] = W_sub
            
            U_full[:, r:] = U_sub
            V_full[6:, r:] = V_sub
            W_full[6:, r:] = W_sub
            
            if verify_full(U_full, V_full, W_full):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = rank_full
                print(f"Found valid decomposition with rank {best_rank}")
                break
        else:
            print(f"Rank {r} for 3x3x2 failed.")

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
