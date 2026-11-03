"""
Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
The output factor follows the official trace convention W[k*n+i,r].

Strategy:
1. Start with the schoolbook decomposition (rank = 2*3*7 = 42).
2. Use Full ALS (Alternating Least Squares) where the linear systems for each factor
   include BOTH valid and spurious constraints. This directly minimizes the violation
   of the tensor structure, rather than just fitting valid entries and hoping spurious
   ones vanish.
3. Search for ranks from low to high, using multiple random initializations.
"""

import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]
N, M, P = DIMENSIONS

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Precompute valid indices and target tensor structure
    # Target T[a, b, c] = 1 if (a,b,c) is a valid (i,j,k) mapping, else 0
    # a = i*M + j
    # b = j*P + k
    # c = k*N + i
    
    # We need t_c vectors for W update, t_a for U update, t_b for V update.
    # t_c is a vector of length N*M*M*P (all a,b pairs) for a fixed c.
    # Actually, the system is A_full * x = t.
    # A_full is (N*M*M*P) x r.
    # Row index = a * (M*P) + b.
    
    # Precompute t vectors for all c, a, b
    # t_W[c, idx(a,b)] = 1 if valid, else 0
    # t_U[a, idx(b,c)] = 1 if valid, else 0
    # t_V[b, idx(a,c)] = 1 if valid, else 0
    
    # Dimensions:
    # a in [0, N*M)
    # b in [0, M*P)
    # c in [0, P*N)
    
    # t_W: shape (P*N, N*M*M*P)
    # t_U: shape (N*M, M*P*P*N)
    # t_V: shape (M*P, N*M*P*N)
    
    # This might be memory heavy?
    # N*M = 6, M*P = 21, P*N = 14.
    # t_W: 14 * (6*21) = 14 * 126 = 1764 floats. Tiny.
    # t_U: 6 * (21*14) = 6 * 294 = 1764 floats. Tiny.
    # t_V: 21 * (6*14) = 21 * 84 = 1764 floats. Tiny.
    
    t_W = np.zeros((P * N, N * M * M * P))
    t_U = np.zeros((N * M, M * P * P * N))
    t_V = np.zeros((M * P, N * M * P * N))
    
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                
                # t_W[c, a * (M*P) + b] = 1
                t_W[c, a * (M * P) + b] = 1.0
                
                # t_U[a, b * (P*N) + c] = 1
                t_U[a, b * (P * N) + c] = 1.0
                
                # t_V[b, a * (P*N) + c] = 1
                t_V[b, a * (P * N) + c] = 1.0

    # Schoolbook decomposition
    def schoolbook():
        rank = N * M * P
        U = np.zeros((N * M, rank))
        V = np.zeros((M * P, rank))
        W = np.zeros((P * N, rank))
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j, col] = 1
                    V[j * P + k, col] = 1
                    W[k * N + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = N * M * P
    
    def check_decomposition_full(U, V, W, tolerance=1e-8):
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        # Check all valid entries
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    val = np.sum(U[i*M+j, :] * V[j*P+k, :] * W[k*N+i, :])
                    if abs(val - 1.0) > tolerance:
                        return False
        
        # Check all spurious entries
        valid_set = set()
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    valid_set.add((i*M+j, j*P+k, k*N+i))
        
        # Vectorized check for spurious entries
        # Construct full tensor? No, too big.
        # Just iterate. 6*21*14 = 1764 entries.
        for a in range(N * M):
            for b in range(M * P):
                for c in range(P * N):
                    if (a, b, c) not in valid_set:
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val) > tolerance:
                            return False
        return True

    def als_refine_full(U, V, W, target_rank, n_iterations=50, time_limit=None):
        """
        Refine U, V, W using Full ALS.
        Each step solves a least squares problem that includes spurious constraints.
        """
        for iteration in range(n_iterations):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Update W
            # A_full shape: (N*M*M*P, target_rank)
            # A_full[idx(a,b), r] = U[a, r] * V[b, r]
            # We can compute this as:
            # U_flat = U.T -> (r, N*M)
            # V_flat = V.T -> (r, M*P)
            # We need outer product structure.
            
            # Efficient way:
            # K = U[:, None, :] * V[None, :, :] -> (N*M, M*P, r)
            # A_full = K.reshape(N*M*M*P, r)
            K_W = U[:, None, :] * V[None, :, :]
            A_full_W = K_W.reshape(N * M * M * P, target_rank)
            
            W_new = np.zeros((P * N, target_rank))
            for c in range(P * N):
                t_c = t_W[c]
                # Solve A_full_W * W_new[c] = t_c
                sol, residuals, rank_, s = np.linalg.lstsq(A_full_W, t_c, rcond=None)
                W_new[c, :] = sol
            W = W_new
            
            # Update U
            # A_full_U[idx(b,c), r] = V[b, r] * W[c, r]
            K_U = V[:, None, :] * W[None, :, :]
            A_full_U = K_U.reshape(M * P * P * N, target_rank)
            
            U_new = np.zeros((N * M, target_rank))
            for a in range(N * M):
                t_a = t_U[a]
                sol, residuals, rank_, s = np.linalg.lstsq(A_full_U, t_a, rcond=None)
                U_new[a, :] = sol
            U = U_new
            
            # Update V
            # A_full_V[idx(a,c), r] = U[a, r] * W[c, r]
            K_V = U[:, None, :] * W[None, :, :]
            A_full_V = K_V.reshape(N * M * P * N, target_rank)
            
            V_new = np.zeros((M * P, target_rank))
            for b in range(M * P):
                t_b = t_V[b]
                sol, residuals, rank_, s = np.linalg.lstsq(A_full_V, t_b, rcond=None)
                V_new[b, :] = sol
            V = V_new
            
        return U, V, W

    # Try to find a lower rank decomposition
    # Start from low rank and go up
    # Schoolbook is 42. Try 30 to 41.
    for target_rank in range(30, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        n_trials = 10
        for trial in range(n_trials):
            if time.time() > deadline:
                break
            
            # Random initialization
            scale = 1.0 / np.sqrt(target_rank)
            U = np.random.randn(N * M, target_rank) * scale
            V = np.random.randn(M * P, target_rank) * scale
            W = np.random.randn(P * N, target_rank) * scale
            
            # Run Full ALS refinement
            U, V, W = als_refine_full(U, V, W, target_rank, n_iterations=20, time_limit=deadline)
            
            # Quick check
            # If ALS converged well, valid entries should be close to 1.
            # Check a few valid entries
            valid_ok = True
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        val = np.sum(U[i*M+j, :] * V[j*P+k, :] * W[k*N+i, :])
                        if abs(val - 1.0) > 1e-6:
                            valid_ok = False
                            break
                    if not valid_ok: break
                if not valid_ok: break
            
            if valid_ok:
                # Full verification
                if check_decomposition_full(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
    
    # Convert to lists for the return format
    U = best_result["U"]
    V = best_result["V"]
    W = best_result["W"]
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
