import time
import numpy as np
from scipy.optimize import least_squares
import itertools

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Trivial decomposition as fallback
    R_trivial = n * m * p
    U_trivial = np.zeros((n * m, R_trivial))
    V_trivial = np.zeros((m * p, R_trivial))
    W_trivial = np.zeros((p * n, R_trivial))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_trivial[i * m + j, r] = 1
                V_trivial[j * p + k, r] = 1
                W_trivial[k * n + i, r] = 1
                
    best_U = U_trivial
    best_V = V_trivial
    best_W = W_trivial
    best_rank = R_trivial

    # Helper to verify a decomposition
    def verify_decomp(U, V, W, n, m, p, tol=1e-9):
        R = U.shape[1]
        # Check condition: sum_r U[i,j,r] * V[j2,k,r] * W[k,i,r] == delta[j, j2]
        # for all i, k, j, j2
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for j2 in range(m):
                        val = 0.0
                        for r in range(R):
                            val += U[i * m + j, r] * V[j2 * p + k, r] * W[k * n + i, r]
                        target = 1.0 if j == j2 else 0.0
                        if abs(val - target) > tol:
                            return False
        return True

    def try_find_decomp(n, m, p, target_rank, time_limit):
        """
        Attempts to find a rank target_rank decomposition for n x m by m x p.
        Returns (U, V, W) if found, else None.
        """
        if time.time() > time_limit:
            return None
            
        # Reshape U, V, W into a single vector for optimization
        # U: n*m x R, V: m*p x R, W: p*n x R
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        size_W = p * n * target_rank
        total_size = size_U + size_V + size_W
        
        # Random initialization
        np.random.seed(42)
        x0 = np.random.randn(total_size) * 0.1
        
        def residuals(x):
            U = x[:size_U].reshape(n * m, target_rank)
            V = x[size_U:size_U + size_V].reshape(m * p, target_rank)
            W = x[size_U + size_V:].reshape(p * n, target_rank)
            
            res = []
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for j2 in range(m):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i * m + j, r] * V[j2 * p + k, r] * W[k * n + i, r]
                            target = 1.0 if j == j2 else 0.0
                            res.append(val - target)
            return np.array(res)
        
        # Use least_squares with multiple restarts
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > time_limit:
                break
                
            if restart == 0:
                x_start = x0
            else:
                x_start = np.random.randn(total_size) * 0.5
                
            try:
                sol = least_squares(residuals, x_start, method='lm', max_nfev=1000)
                if sol.cost < 1e-10:
                    x_sol = sol.x
                    U = x_sol[:size_U].reshape(n * m, target_rank)
                    V = x_sol[size_U:size_U + size_V].reshape(m * p, target_rank)
                    W = x_sol[size_U + size_V:].reshape(p * n, target_rank)
                    
                    # Try to snap to half-integers
                    U_half = np.round(U * 2) / 2
                    V_half = np.round(V * 2) / 2
                    W_half = np.round(W * 2) / 2
                    
                    if verify_decomp(U_half, V_half, W_half, n, m, p, tol=1e-6):
                        return U_half, V_half, W_half
                    else:
                        # If snapping fails, check if the float solution is valid (might not meet integer/half-integer req)
                        # The prompt requires integer/half-integer. So we must snap.
                        # If snap fails, we might try to keep floats if the evaluator accepts? 
                        # "integer/half-integer entries" is explicit.
                        # So we must return None if we can't snap.
                        pass
                        
            except Exception:
                continue
                
        return None

    # Strategy: Split 3x7 into 3x3 and 3x4
    # Find decomp for 2x3x3 and 2x3x4
    
    # Try 2x3x3
    U_233 = None
    V_233 = None
    W_233 = None
    rank_233 = 18 # Trivial
    
    for r in range(12, 18):
        if time.time() > deadline - 30: break
        res = try_find_decomp(2, 3, 3, r, deadline - 30)
        if res:
            U_233, V_233, W_233 = res
            rank_233 = r
            break

    # Try 2x3x4
    U_234 = None
    V_234 = None
    W_234 = None
    rank_234 = 24 # Trivial
    
    for r in range(16, 24):
        if time.time() > deadline - 10: break
        res = try_find_decomp(2, 3, 4, r, deadline - 10)
        if res:
            U_234, V_234, W_234 = res
            rank_234 = r
            break

    # Combine
    if U_233 is not None and U_234 is not None:
        R_total = rank_233 + rank_234
        
        U_total = np.zeros((2 * 3, R_total))
        V_total = np.zeros((3 * 7, R_total))
        W_total = np.zeros((7 * 2, R_total))
        
        # Fill 2x3x3 part (columns 0..rank_233-1)
        # U_233 is 6 x R_233
        # V_233 is 9 x R_233 (indices j*3+k, k in 0..2)
        # W_233 is 6 x R_233 (indices k*2+i, k in 0..2)
        
        U_total[:, :rank_233] = U_233
        
        # Map V_233 (3x3) to V_total (3x7)
        # V_233[j, k] goes to V_total[j*7 + k]
        for j in range(3):
            for k in range(3):
                V_total[j * 7 + k, :rank_233] = V_233[j * 3 + k, :]
                
        # Map W_233 (3x2) to W_total (7x2)
        # W_233[k, i] goes to W_total[k*2 + i]
        for k in range(3):
            for i in range(2):
                W_total[k * 2 + i, :rank_233] = W_233[k * 2 + i, :]
                
        # Fill 2x3x4 part (columns rank_233..R_total-1)
        # U_234 is 6 x R_234
        # V_234 is 12 x R_234 (indices j*4+k, k in 0..3)
        # W_234 is 8 x R_234 (indices k*2+i, k in 0..3)
        
        U_total[:, rank_233:] = U_234
        
        # Map V_234 (3x4) to V_total (3x7)
        # The 3x4 block corresponds to columns 3..6 of the 3x7 matrix
        # So V_234[j, k] (k in 0..3) goes to V_total[j*7 + (3+k)]
        for j in range(3):
            for k in range(4):
                V_total[j * 7 + (3 + k), rank_233:] = V_234[j * 4 + k, :]
                
        # Map W_234 (4x2) to W_total (7x2)
        # W_234[k, i] (k in 0..3) goes to W_total[(3+k)*2 + i]
        for k in range(4):
            for i in range(2):
                W_total[(3 + k) * 2 + i, rank_233:] = W_234[k * 2 + i, :]
                
        # Verify combined
        if verify_decomp(U_total, V_total, W_total, 2, 3, 7, tol=1e-6):
            if R_total < best_rank:
                best_U = U_total
                best_V = V_total
                best_W = W_total
                best_rank = R_total
    else:
        # Fallback to trivial
        best_U = U_trivial
        best_V = V_trivial
        best_W = W_trivial
        best_rank = R_trivial

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
