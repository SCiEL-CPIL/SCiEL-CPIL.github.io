import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 18
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    # Function to verify a decomposition with exact rational arithmetic
    def verify_exact(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        r = U.shape[1]
        # Convert to Fraction for exactness
        U_frac = np.array([[Fraction(U[i, c]).limit_denominator(10**6) for c in range(r)] for i in range(n*m)], dtype=object)
        V_frac = np.array([[Fraction(V[j, c]).limit_denominator(10**6) for c in range(r)] for j in range(m*p)], dtype=object)
        W_frac = np.array([[Fraction(W[k, c]).limit_denominator(10**6) for c in range(r)] for k in range(p*n)], dtype=object)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += U_frac[i*m+j, c] * V_frac[j*p+k, c] * W_frac[k*n+i, c]
                    if s != Fraction(1):
                        return False
        return True
    
    # Function to verify with floating point (fast)
    def verify_float(U, V, W, atol=1e-8):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > atol:
                        return False
        return True
    
    # Schoolbook decomposition
    def schoolbook():
        rank = rank_schoolbook
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = None, None, None
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Known lower bounds: For 2x3x3, the minimum rank is at least 17
    # (It's a classic open problem whether it's 17 or 18 for 2x2x2, 
    #  but for 2x3x3, schoolbook is 18, and lower bounds are typically lower)
    
    # Actually, let's just try to find rank 17.
    # If we can't, return schoolbook.
    
    # Strategy: Use ALS (Alternating Least Squares) to find a low-rank decomposition
    # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    def try_rank(target_rank, max_iterations=1000, max_restarts=20):
        nonlocal best_U, best_V, best_W, best_rank, start_time, deadline
        
        if target_rank >= best_rank:
            return
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return
            
            # Initialize with schoolbook + noise
            U_sb, V_sb, W_sb = schoolbook()
            
            # Use first target_rank columns of schoolbook
            U = U_sb[:, :target_rank].copy().astype(float)
            V = V_sb[:, :target_rank].copy().astype(float)
            W = W_sb[:, :target_rank].copy().astype(float)
            
            # Add noise
            noise_level = 0.1
            U += noise_level * np.random.randn(n*m, target_rank)
            V += noise_level * np.random.randn(m*p, target_rank)
            W += noise_level * np.random.randn(p*n, target_rank)
            
            # ALS iterations
            prev_err = float('inf')
            for it in range(max_iterations):
                if time.time() > deadline:
                    return
                
                # Update U: For each (i,j), U[i*m+j, :] should satisfy
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # This is a linear system: U[i*m+j, :] * (V[j*p+k,:] * W[k*n+i,:])_c = 1
                # For fixed i,j: U[i*m+j, c] = sum over k of ... 
                # Actually: For fixed i,j, we need:
                # sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
                # This is: U[i*m+j, :] * M = 1_vec where M[k,c] = V[j*p+k,c] * W[k*n+i,c]
                # So U[i*m+j, :] = M^+ 1_vec (pseudoinverse)
                
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            M[k, :] = V[j*p+k] * W[k*n+i]
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V: For each (j,k), V[j*p+k, :]
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
                # V[j*p+k, :] * N = 1_vec where N[i,c] = U[i*m+j,c] * W[k*n+i,c]
                for j in range(m):
                    for k in range(p):
                        N = np.zeros((n, target_rank))
                        for i in range(n):
                            N[i, :] = U[i*m+j] * W[k*n+i]
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(N, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W: For each (k,i), W[k*n+i, :]
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
                # W[k*n+i, :] * P = 1_vec where P[j,c] = U[i*m+j,c] * V[j*p+k,c]
                for k in range(p):
                    for i in range(n):
                        P = np.zeros((m, target_rank))
                        for j in range(m):
                            P[j, :] = U[i*m+j] * V[j*p+k]
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(P, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                            err += (s - 1.0)**2
                err = np.sqrt(err)
                
                if err < prev_err - 1e-12:
                    prev_err = err
                else:
                    if err < 1e-6:
                        # Good enough, try to round
                        U_round = np.round(U * 2) / 2
                        V_round = np.round(V * 2) / 2
                        W_round = np.round(W * 2) / 2
                        
                        if verify_float(U_round, V_round, W_round, atol=1e-8):
                            # Verify with exact arithmetic
                            if verify_exact(U_round, V_round, W_round):
                                best_U, best_V, best_W = U_round, V_round, W_round
                                best_rank = target_rank
                                return
                        break
                    break
                
                if err < 1e-10:
                    break
        
        return
    
    # Try ranks from 17 down to 10 (be conservative)
    # For 2x3x3, known lower bound is around 17
    for target_rank in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline:
            break
        try_rank(target_rank, max_iterations=500, max_restarts=10)
        if best_rank < rank_schoolbook:
            break
    
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        U, V, W = schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
