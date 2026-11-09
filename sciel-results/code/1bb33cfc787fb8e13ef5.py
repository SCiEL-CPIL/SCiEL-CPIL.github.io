"""Matrix multiplication 2,4,5 — rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill
    
    def make_schoolbook():
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
    
    def verify_exact(U_list, V_list, W_list, R):
        """Check if decomposition is exact: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    def try_rank_als(R, max_iters=100, n_restarts=8, seed_base=0):
        """Try to find exact decomposition of rank R using ALS-like refinement"""
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            rng = np.random.RandomState(seed_base + restart * 17 + 42)
            
            # Try different initialization strategies
            if restart % 3 == 0:
                U = rng.randn(n*m, R) * 0.5
                V = rng.randn(m*p, R) * 0.5
                W = rng.randn(p*n, R) * 0.5
            elif restart % 3 == 1:
                U = rng.randn(n*m, R)
                V = rng.randn(m*p, R)
                W = rng.randn(p*n, R)
                # Add some structure: make first few columns sparse
                for r in range(min(3, R)):
                    U[:, r] = 0
                    U[rng.choice(n*m, size=min(R//4, n*m), replace=False), r] = 1
                    V[:, r] = 0
                    V[rng.choice(m*p, size=min(R//4, m*p), replace=False), r] = 1
                    W[:, r] = 0
                    W[rng.choice(p*n, size=min(R//4, p*n), replace=False), r] = 1
            else:
                U = rng.randn(n*m, R)
                V = rng.randn(m*p, R)
                W = rng.randn(p*n, R)
                # Scale to avoid overflow
                U /= np.linalg.norm(U)
                V /= np.linalg.norm(V)
                W /= np.linalg.norm(W)
            
            best_err = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Update U: for each row (i,j), solve M^T @ U_row = ones(p)
                for i in range(n):
                    for j in range(m):
                        # Build matrix M of shape (p, R): M[k,r] = V[j*p+k,r] * W[k*n+i,r]
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each row (j,k), solve N^T @ V_row = ones(n)
                for j in range(m):
                    for k in range(p):
                        N = np.zeros((n, R))
                        for i in range(n):
                            N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(N.T, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each row (k,i), solve P^T @ W_row = ones(m)
                for k in range(p):
                    for i in range(n):
                        P = np.zeros((m, R))
                        for j in range(m):
                            P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(P.T, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err += (s - 1.0) ** 2
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if err < 1e-14:
                        # Found near-exact solution, try to convert to fractions
                        # Use higher precision denominator
                        U_frac = [[Fraction(U[i*m+j, r]).limit_denominator(10**9) for r in range(R)] for i in range(n) for j in range(m)]
                        V_frac = [[Fraction(V[j*p+k, r]).limit_denominator(10**9) for r in range(R)] for j in range(m) for k in range(p)]
                        W_frac = [[Fraction(W[k*n+i, r]).limit_denominator(10**9) for r in range(R)] for k in range(p) for i in range(n)]
                        
                        if verify_exact(U_frac, V_frac, W_frac, R):
                            return {"U": U_frac, "V": V_frac, "W": W_frac}
                
                if time.time() > deadline:
                    return None
            
            # After all iterations, try to polish the best solution
            if best_err < 1e-10:
                # Do a final high-precision pass
                U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
                for _ in range(50):
                    if time.time() > deadline:
                        return None
                    for i in range(n):
                        for j in range(m):
                            M = np.zeros((p, R))
                            for k in range(p):
                                M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                            try:
                                U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                            except:
                                pass
                    for j in range(m):
                        for k in range(p):
                            N = np.zeros((n, R))
                            for i in range(n):
                                N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                            try:
                                V[j*p+k, :] = np.linalg.lstsq(N.T, np.ones(n), rcond=None)[0]
                            except:
                                pass
                    for k in range(p):
                        for i in range(n):
                            P = np.zeros((m, R))
                            for j in range(m):
                                P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                            try:
                                W[k*n+i, :] = np.linalg.lstsq(P.T, np.ones(m), rcond=None)[0]
                            except:
                                pass
                    
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for r in range(R):
                                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                                err += (s - 1.0) ** 2
                    
                    if err < 1e-14:
                        U_frac = [[Fraction(U[i*m+j, r]).limit_denominator(10**9) for r in range(R)] for i in range(n) for j in range(m)]
                        V_frac = [[Fraction(V[j*p+k, r]).limit_denominator(10**9) for r in range(R)] for j in range(m) for k in range(p)]
                        W_frac = [[Fraction(W[k*n+i, r]).limit_denominator(10**9) for r in range(R)] for k in range(p) for i in range(n)]
                        if verify_exact(U_frac, V_frac, W_frac, R):
                            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Try ranks from 21 up to 39 (below schoolbook 40)
    # Use more restarts for smaller ranks, fewer for larger
    for R in range(21, 40):
        if time.time() > deadline:
            break
        # Allocate more time to smaller ranks
        if R <= 25:
            n_restarts = 12
            max_iters = 80
        elif R <= 30:
            n_restarts = 8
            max_iters = 60
        else:
            n_restarts = 5
            max_iters = 40
        result = try_rank_als(R, max_iters=max_iters, n_restarts=n_restarts, seed_base=R*100)
        if result is not None:
            return result
    
    # Fallback to Schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END
