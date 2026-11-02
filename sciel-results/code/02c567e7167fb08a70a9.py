"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import numpy as np
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    U_dim = n * m  # 8
    V_dim = m * p  # 24
    W_dim = p * n  # 12
    
    # Build the target tensor T of shape (U_dim, V_dim, W_dim)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(U_dim)]
    V_school = [[0] * rank_school for _ in range(V_dim)]
    W_school = [[0] * rank_school for _ in range(W_dim)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school, "rank": rank_school}
    
    def verify(U_mat, V_mat, W_mat, tol=1e-10):
        """Check if U,V,W exactly decompose T (within tol)."""
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Vectorized check: for each c, T[:,:,c] should equal sum_r (U[:,r]*V[:,r]) * W[r,c]
        # Let's do it efficiently
        for c in range(W_dim):
            # Compute sum_r U[:,r] * V[:,r] * W[r,c]
            # This is a sum of outer products scaled by W[r,c]
            recon = np.zeros((U_dim, V_dim), dtype=np.float64)
            for r in range(U_mat.shape[1]):
                recon += U_mat[:, r] * V_mat[:, r] * W_mat[r, c]
            if np.max(np.abs(recon - T[:, :, c])) > tol:
                return False
        return True
    
    def round_to_rational(U_mat, V_mat, W_mat, max_denom=8):
        """Try to round values to simple rationals and verify."""
        U_r = np.zeros_like(U_mat)
        V_r = np.zeros_like(V_mat)
        W_r = np.zeros_like(W_mat)
        
        for idx in range(U_mat.size):
            val = U_mat.flat[idx]
            # Try rounding to nearest 1/max_denom
            best_val = 0.0
            best_err = float('inf')
            for d in range(1, max_denom + 1):
                num = round(val * d)
                err = abs(val - num / d)
                if err < best_err:
                    best_err = err
                    best_val = num / d
            U_r.flat[idx] = best_val
            
        for idx in range(V_mat.size):
            val = V_mat.flat[idx]
            best_val = 0.0
            best_err = float('inf')
            for d in range(1, max_denom + 1):
                num = round(val * d)
                err = abs(val - num / d)
                if err < best_err:
                    best_err = err
                    best_val = num / d
            V_r.flat[idx] = best_val
            
        for idx in range(W_mat.size):
            val = W_mat.flat[idx]
            best_val = 0.0
            best_err = float('inf')
            for d in range(1, max_denom + 1):
                num = round(val * d)
                err = abs(val - num / d)
                if err < best_err:
                    best_err = err
                    best_val = num / d
            W_r.flat[idx] = best_val
            
        if verify(U_r, V_r, W_r, tol=1e-9):
            return U_r, V_r, W_r
        return None
    
    def try_rank(r, max_iters=2000, seeds=5):
        """Try to find a rank-r decomposition using ALS with random restarts."""
        if time.time() > deadline - 5:
            return None
        
        for seed in range(seeds):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(seed * 1000 + r)
            U_mat = np.random.randn(U_dim, r) * 0.5
            V_mat = np.random.randn(V_dim, r) * 0.5
            W_mat = np.random.randn(W_dim, r) * 0.5
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                # Fix U and V, solve for W
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For each c, this is a linear system in W[:,c]
                # Let's build the system:
                # For each (a,b), we have an equation: T[a,b,c] = sum_r (U[a,r]*V[b,r]) * W[c,r]
                # This is: for each c, W[:,c] is a vector of length r
                # The coefficient matrix for W[:,c] has entries (a,b,r) -> U[a,r]*V[b,r]
                
                # Let's vectorize: create a matrix M of shape (U_dim*V_dim, r)
                # where M[(a,b), r] = U[a,r] * V[b,r]
                # Then for each c: M * W[:,c] = T[:, :, c].ravel()
                
                M = np.zeros((U_dim * V_dim, r), dtype=np.float64)
                for r_idx in range(r):
                    M[:, r_idx] = (U_mat[:, r_idx].reshape(U_dim, 1) * V_mat[:, r_idx].reshape(1, V_dim)).ravel()
                
                # Solve least squares for each c
                W_new = np.zeros((W_dim, r), dtype=np.float64)
                for c in range(W_dim):
                    rhs = T[:, :, c].ravel()
                    # M is (U_dim*V_dim, r), we want W[:,c] such that M @ W[:,c] ≈ rhs
                    # This is an underdetermined system typically, use lstsq
                    try:
                        sol, residuals, rank_val, sv = np.linalg.lstsq(M, rhs, rcond=None)
                        W_new[:, c] = sol
                    except:
                        return None
                
                # Fix V and W, solve for U
                M2 = np.zeros((U_dim, r), dtype=np.float64)
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For each a: T[a,:,:].ravel() = sum_r U[a,r] * (V[:,r] * W[:,r]).ravel()
                # Let N2 be (r, V_dim*W_dim) where N2[r, (b,c)] = V[b,r] * W[c,r]
                # Then T[a,:,:].ravel() = sum_r U[a,r] * N2[r, :]
                # So U[a,:] is a row, and we have: U[a,:] @ N2.T = T[a,:,:].ravel()
                
                N2 = np.zeros((r, V_dim * W_dim), dtype=np.float64)
                for r_idx in range(r):
                    N2[r_idx, :] = (V_mat[:, r_idx].reshape(V_dim, 1) * W_mat[:, r_idx].reshape(1, W_dim)).ravel()
                
                U_new = np.zeros((U_dim, r), dtype=np.float64)
                for a in range(U_dim):
                    rhs = T[a, :, :].ravel()
                    # U[a,:] @ N2.T = rhs => N2 @ U[a,:].T = rhs
                    try:
                        sol, residuals, rank_val, sv = np.linalg.lstsq(N2, rhs, rcond=None)
                        U_new[a, :] = sol
                    except:
                        return None
                
                # Fix U and W, solve for V
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For each b: T[:,b,:].ravel() = sum_r V[b,r] * (U[:,r] * W[:,r]).ravel()
                # Let N3 be (r, U_dim*W_dim) where N3[r, (a,c)] = U[a,r] * W[c,r]
                # Then T[:,b,:].ravel() = sum_r V[b,r] * N3[r, :]
                
                N3 = np.zeros((r, U_dim * W_dim), dtype=np.float64)
                for r_idx in range(r):
                    N3[r_idx, :] = (U_new[:, r_idx].reshape(U_dim, 1) * W_mat[:, r_idx].reshape(1, W_dim)).ravel()
                
                V_new = np.zeros((V_dim, r), dtype=np.float64)
                for b in range(V_dim):
                    rhs = T[:, b, :].ravel()
                    try:
                        sol, residuals, rank_val, sv = np.linalg.lstsq(N3, rhs, rcond=None)
                        V_new[b, :] = sol
                    except:
                        return None
                
                # Check convergence
                diff_U = np.max(np.abs(U_new - U_mat))
                diff_V = np.max(np.abs(V_new - V_mat))
                diff_W = np.max(np.abs(W_new - W_mat))
                
                U_mat = U_new
                V_mat = V_new
                W_mat = W_new
                
                if diff_U + diff_V + diff_W < 1e-8:
                    break
                
                # Verify
                if verify(U_mat, V_mat, W_mat, tol=1e-6):
                    # Try to round
                    rounded = round_to_rational(U_mat, V_mat, W_mat, max_denom=16)
                    if rounded:
                        Ur, Vr, Wr = rounded
                        U_list = [[float(Ur[idx, c]) for c in range(r)] for idx in range(U_dim)]
                        V_list = [[float(Vr[idx, c]) for c in range(r)] for idx in range(V_dim)]
                        W_list = [[float(Wr[idx, c]) for c in range(r)] for idx in range(W_dim)]
                        return {"U": U_list, "V": V_list, "W": W_list, "rank": r}
                    else:
                        # Return unrounded if verify passes with tight tol
                        if verify(U_mat, V_mat, W_mat, tol=1e-10):
                            U_list = [[float(U_mat[idx, c]) for c in range(r)] for idx in range(U_dim)]
                            V_list = [[float(V_mat[idx, c]) for c in range(r)] for idx in range(V_dim)]
                            W_list = [[float(W_mat[idx, c]) for c in range(r)] for idx in range(W_dim)]
                            return {"U": U_list, "V": V_list, "W": W_list, "rank": r}
            
            # If we didn't converge to a verified solution, try rounding the final iterate
            if verify(U_mat, V_mat, W_mat, tol=1e-5):
                rounded = round_to_rational(U_mat, V_mat, W_mat, max_denom=32)
                if rounded:
                    Ur, Vr, Wr = rounded
                    U_list = [[float(Ur[idx, c]) for c in range(r)] for idx in range(U_dim)]
                    V_list = [[float(Vr[idx, c]) for c in range(r)] for idx in range(V_dim)]
                    W_list = [[float(Wr[idx, c]) for c in range(r)] for idx in range(W_dim)]
                    return {"U": U_list, "V": V_list, "W": W_list, "rank": r}
        
        return None
    
    # Try ranks from low to high
    # Known lower bounds suggest rank >= 24 for (2,4,6)
    # Let's try from 24 up to schoolbook
    for target_rank in range(24, rank_school + 1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(target_rank, max_iters=500, seeds=3)
        if result is not None:
            best = result
            break
    
    # If we found something better, return it
    if best["rank"] < rank_school:
        return {"U": best["U"], "V": best["V"], "W": best["W"]}
    
    # Otherwise return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
