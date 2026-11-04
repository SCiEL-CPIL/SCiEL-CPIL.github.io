import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify_exact(U, V, W, rank):
        """Verify exact decomposition with half-integer entries using Fractions"""
        # Convert to Fractions for exact check
        U_frac = [[Fraction(val).limit_denominator(1000) for val in row] for row in U]
        V_frac = [[Fraction(val).limit_denominator(1000) for val in row] for row in V]
        W_frac = [[Fraction(val).limit_denominator(1000) for val in row] for row in W]
        
        # Check if all entries are half-integers (denominator 1 or 2)
        for mat in [U_frac, V_frac, W_frac]:
            for row in mat:
                for val in row:
                    if val.denominator not in [1, 2]:
                        return False
        
        # Check tensor equation
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U_frac[i*m+j][c] * V_frac[j*p+k][c] * W_frac[k*n+i][c]
                    if s != 1:
                        return False
        return True

    def try_rank(r, max_iters=500, seed=42):
        """Try to find rank-r decomposition using vectorized ALS"""
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n*m, r)
        V = rng.randn(m*p, r)
        W = rng.randn(p*n, r)
        
        # Precompute index mappings for speed
        # For U update: for each (i,j), we need V[j*p+k,:] and W[k*n+i,:]
        # For V update: for each (j,k), we need U[i*m+j,:] and W[k*n+i,:]
        # For W update: for each (k,i), we need U[i*m+j,:] and V[j*p+k,:]
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve U[i*m+j,:] @ M.T = ones(p)
                    # U[i*m+j,:] = ones(p) @ pinv(M.T)
                    # Using lstsq for stability
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence and rounding
            if it % 20 == 0:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                            err += (s - 1.0)**2
                if err < 1e-10:
                    # Try rounding to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    # Verify with exact arithmetic
                    if verify_exact(U_r.tolist(), V_r.tolist(), W_r.tolist(), r):
                        return (U_r.tolist(), V_r.tolist(), W_r.tolist(), r)
        
        return None
    
    # Try decreasing ranks
    # Start from 83 down to 43
    for r in range(best_rank - 1, 42, -1):
        if time.time() > deadline - 10:
            break
        # Try a few seeds for each rank
        for seed in [42, 123, 999]:
            if time.time() > deadline - 10:
                break
            result = try_rank(r, max_iters=200, seed=seed)
            if result is not None:
                U, V, W, rank = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
        if best_rank < best_rank: # If found, break outer loop? No, we want min rank.
             # Actually, if we find a rank r, we can stop if we are going down.
             # But maybe we can find r-1?
             # Given time limits, finding the first valid lower rank is a good win.
             pass
    
    return best_result
