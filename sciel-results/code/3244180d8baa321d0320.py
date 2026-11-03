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
    
    # Target rank to try for the full 3x3x4 problem
    # We will try to find a decomposition for 3x3x2 with rank r, then use 2*r for 3x3x4.
    # Schoolbook 3x3x2 is 18. Full schoolbook is 36.
    # We try r = 16, 17, 18.
    
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
        # Check zero entries (sample or full check if fast enough)
        # Full check: 9*12*12 = 1296 entries.
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
        """
        Search for a rank decomposition of 3x3x2 matrix multiplication.
        Returns (U, V, W) or None if not found within time.
        U: (9, rank_332), V: (6, rank_332), W: (6, rank_332)
        """
        n_sub, m_sub, p_sub = 3, 3, 2
        # Target tensor T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) for i in 0..2, j in 0..2, k in 0..1
        # a in 0..8, b in 0..5, c in 0..5
        
        # Flatten variables
        # U: 9 * rank, V: 6 * rank, W: 6 * rank
        n_vars = 9 * rank_332 + 6 * rank_332 + 6 * rank_332
        x0 = np.random.randn(n_vars) * 0.1
        
        def unpack(x):
            U = x[:9*rank_332].reshape(9, rank_332)
            V = x[9*rank_332:15*rank_332].reshape(6, rank_332)
            W = x[15*rank_332:21*rank_332].reshape(6, rank_332)
            return U, V, W

        def error(x):
            U, V, W = unpack(x)
            err = 0.0
            # Precompute products for speed?
            # For each a, b, c:
            # sum_r U[a,r] V[b,r] W[c,r]
            # Vectorized:
            # T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            # This creates a 9x6x6 array. 324 entries.
            # Very fast.
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            
            # Target T
            # T[i,j, j,k, k,i] = 1
            # a = i*3+j, b = j*2+k, c = k*3+i
            # Note: c index in 3x3x2 is k*3+i.
            # k in 0..1, i in 0..2. c in 0..5.
            
            # Initialize target to 0
            T_target = np.zeros((9, 6, 6))
            for i in range(3):
                for j in range(3):
                    for k in range(2):
                        a = i * 3 + j
                        b = j * 2 + k
                        c = k * 3 + i
                        T_target[a, b, c] = 1.0
            
            diff = T_approx - T_target
            return np.sum(diff**2)

        # Optimize
        result = scipy.optimize.minimize(error, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-8})
        
        if result.fun < 1e-8:
            U, V, W = unpack(result.x)
            # Verify exactness with higher precision check
            # Check if entries are close to half-integers? Not required by prompt, just exact decomposition.
            # But integer/half-integer is preferred.
            # We can round to nearest 0.5 and re-verify.
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            # Check if rounded version is valid
            T_approx_r = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
            diff_r = T_approx_r - T_target
            if np.sum(diff_r**2) < 1e-8:
                return U_round, V_round, W_round
            else:
                # If rounding fails, return the float version if it's exact enough?
                # Prompt says "integer/half-integer entries".
                # So we must return rounded if it works. If not, maybe try to snap more aggressively?
                # For now, if rounded fails, return None.
                return None
        return None

    # Try ranks for 3x3x2
    for r in [16, 17, 18]:
        if time.time() > deadline - 20:
            break
            
        print(f"Trying rank {r} for 3x3x2...")
        result_332 = search_332(r)
        
        if result_332 is not None:
            U_sub, V_sub, W_sub = result_332
            rank_full = 2 * r
            
            # Construct full 3x3x4 matrices
            U_full = np.zeros((9, rank_full))
            V_full = np.zeros((12, rank_full))
            W_full = np.zeros((12, rank_full))
            
            # First half (columns 0..3 of B, rows 0..5 of C)
            U_full[:, :r] = U_sub
            V_full[:6, :r] = V_sub
            W_full[:6, :r] = W_sub
            
            # Second half (columns 4..7 of B, rows 6..11 of C)
            U_full[:, r:] = U_sub
            V_full[6:, r:] = V_sub
            W_full[6:, r:] = W_sub
            
            # Verify
            if verify_full(U_full, V_full, W_full):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = rank_full
                print(f"Found valid decomposition with rank {best_rank}")
                break
        else:
            print(f"Rank {r} for 3x3x2 failed.")

    # If no better solution found, return schoolbook
    if best_rank == 36:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
