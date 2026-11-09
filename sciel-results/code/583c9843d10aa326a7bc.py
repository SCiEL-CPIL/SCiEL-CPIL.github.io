"""Matrix multiplication 2,4,5 — rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin before 120s kill
    
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
    
    def try_rank_als(R, max_iters=200, n_restarts=15, seed_base=0):
        """Try to find exact decomposition of rank R using vectorized ALS"""
        for restart in range(n_restarts):
            if time.time() > deadline:
                return None
            rng = np.random.RandomState(seed_base + restart * 17 + 42)
            
            # Different initialization strategies
            if restart % 5 == 0:
                U = rng.randn(n*m, R) * 0.3
                V = rng.randn(m*p, R) * 0.3
                W = rng.randn(p*n, R) * 0.3
            elif restart % 5 == 1:
                U = rng.randn(n*m, R)
                V = rng.randn(m*p, R)
                W = rng.randn(p*n, R)
            elif restart % 5 == 2:
                # Sparse initialization
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                for r in range(R):
                    nz_u = rng.choice(n*m, size=max(1, R//5), replace=False)
                    nz_v = rng.choice(m*p, size=max(1, R//5), replace=False)
                    nz_w = rng.choice(p*n, size=max(1, R//5), replace=False)
                    U[nz_u, r] = rng.randn(len(nz_u))
                    V[nz_v, r] = rng.randn(len(nz_v))
                    W[nz_w, r] = rng.randn(len(nz_w))
            elif restart % 5 == 3:
                # Structured: initialize from schoolbook pattern with perturbation
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r = (i * m + j) * p + k
                            if r < R:
                                U[i*m+j, r] = 1 + 0.1 * rng.randn()
                                V[j*p+k, r] = 1 + 0.1 * rng.randn()
                                W[k*n+i, r] = 1 + 0.1 * rng.randn()
            else:
                U = rng.randn(n*m, R) * 0.5
                V = rng.randn(m*p, R) * 0.5
                W = rng.randn(p*n, R) * 0.5
                # Add small random structure
                for r in range(min(2, R)):
                    U[rng.choice(n*m, size=2, replace=False), r] = 1
                    V[rng.choice(m*p, size=2, replace=False), r] = 1
                    W[rng.choice(p*n, size=2, replace=False), r] = 1
            
            best_err = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Vectorized ALS update for U
                for i in range(n):
                    for j in range(m):
                        # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve M^T @ x = ones(p)
                        try:
                            A = M.T @ M
                            b = M.T @ np.ones(p)
                            U[i*m+j, :] = np.linalg.solve(A + 1e-10*np.eye(R), b)
                        except:
                            pass
                
                # Vectorized ALS update for V
                for j in range(m):
                    for k in range(p):
                        N = np.zeros((n, R))
                        for i in range(n):
                            N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            A = N.T @ N
                            b = N.T @ np.ones(n)
                            V[j*p+k, :] = np.linalg.solve(A + 1e-10*np.eye(R), b)
                        except:
                            pass
                
                # Vectorized ALS update for W
                for k in range(p):
                    for i in range(n):
                        P = np.zeros((m, R))
                        for j in range(m):
                            P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            A = P.T @ P
                            b = P.T @ np.ones(m)
                            W[k*n+i, :] = np.linalg.solve(A + 1e-10*np.eye(R), b)
                        except:
                            pass
                
                # Compute error (vectorized)
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err += (s - 1.0) ** 2
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if err < 1e-12:
                        # Try to convert to fractions
                        U_frac = [[Fraction(float(U[i*m+j, r])).limit_denominator(10**6) for r in range(R)] for i in range(n) for j in range(m)]
                        V_frac = [[Fraction(float(V[j*p+k, r])).limit_denominator(10**6) for r in range(R)] for j in range(m) for k in range(p)]
                        W_frac = [[Fraction(float(W[k*n+i, r])).limit_denominator(10**6) for r in range(R)] for k in range(p) for i in range(n)]
                        
                        if verify_exact(U_frac, V_frac, W_frac, R):
                            return {"U": U_frac, "V": V_frac, "W": W_frac}
                
                # Early stopping if error not improving
                if it > 20 and err > best_err * 1.01:
                    break
            
            # Final polish with best solution
            if best_err < 1e-8:
                U, V, W = best_U.copy(), best_V.copy(), best_W.copy()
                for _ in range(30):
                    if time.time() > deadline:
                        return None
                    for i in range(n):
                        for j in range(m):
                            M = np.zeros((p, R))
                            for k in range(p):
                                M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                            try:
                                A = M.T @ M
                                b = M.T @ np.ones(p)
                                U[i*m+j, :] = np.linalg.solve(A + 1e-10*np.eye(R), b)
                            except:
                                pass
                    for j in range(m):
                        for k in range(p):
                            N = np.zeros((n, R))
                            for i in range(n):
                                N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                            try:
                                A = N.T @ N
                                b = N.T @ np.ones(n)
                                V[j*p+k, :] = np.linalg.solve(A + 1e-10*np.eye(R), b)
                            except:
                                pass
                    for k in range(p):
                        for i in range(n):
                            P = np.zeros((m, R))
                            for j in range(m):
                                P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                            try:
                                A = P.T @ P
                                b = P.T @ np.ones(m)
                                W[k*n+i, :] = np.linalg.solve(A + 1e-10*np.eye(R), b)
                            except:
                                pass
                    
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                err += (s - 1.0) ** 2
                    
                    if err < 1e-12:
                        U_frac = [[Fraction(float(U[i*m+j, r])).limit_denominator(10**6) for r in range(R)] for i in range(n) for j in range(m)]
                        V_frac = [[Fraction(float(V[j*p+k, r])).limit_denominator(10**6) for r in range(R)] for j in range(m) for k in range(p)]
                        W_frac = [[Fraction(float(W[k*n+i, r])).limit_denominator(10**6) for r in range(R)] for k in range(p) for i in range(n)]
                        
                        if verify_exact(U_frac, V_frac, W_frac, R):
                            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Try ranks from 21 up to 39 (below schoolbook 40)
    for R in range(21, 40):
        if time.time() > deadline:
            break
        # Allocate time based on rank
        if R <= 25:
            n_restarts = 20
            max_iters = 150
        elif R <= 30:
            n_restarts = 15
            max_iters = 100
        else:
            n_restarts = 10
            max_iters = 80
        result = try_rank_als(R, max_iters=max_iters, n_restarts=n_restarts, seed_base=R*100)
        if result is not None:
            return result
    
    # Fallback to Schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END
