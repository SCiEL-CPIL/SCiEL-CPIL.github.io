"""Find rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (6, rank)
    # W shape: (p*n, rank) = (6, rank)
    # Tensor T[a,b,c] where a in [0,4), b in [0,6), c in [0,6)
    # T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # i.e., a = i*2+j, b = j*3+k, c = k*2+i
    # T[a,b,c] = sum over (i,j,k) such that a=i*2+j, b=j*3+k, c=k*2+i of 1
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Try to find low-rank decomposition
    # For rank r, we need U (4xr), V (6xr), W (6xr) such that
    # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # Start with schoolbook rank 12
    rank_12 = n * m * p
    U12 = np.zeros((n*m, rank_12))
    V12 = np.zeros((m*p, rank_12))
    W12 = np.zeros((p*n, rank_12))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U12[i*m+j, col] = 1
                V12[j*p+k, col] = 1
                W12[k*n+i, col] = 1
                col += 1
    
    # Try rank 11, 10, 9, 8 by solving linear systems
    # For a given rank r, fix U and V, solve for W, or use iterative approach
    
    best_result = {"U": U12.tolist(), "V": V12.tolist(), "W": W12.tolist()}
    best_rank = rank_12
    
    # Try to reduce rank by attempting rank r decompositions
    # Use a simple approach: for each candidate rank, try random initializations
    # and solve via linear least squares iteratively
    
    deadline = time.time() + 100  # safety margin before 120s limit
    
    for target_rank in range(11, 5, -1):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Iterative solving: fix two, solve for third
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Solve for W given U, V: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # For each c, T[:,:,c] = sum_r W[c,r] * (U*V^T)_r[:,:,...]
                # Actually: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # Reshape: for fixed c, T[:,:,c] = sum_r W[c,r] * (U[:,r] ⊗ V[:,r])
                
                # Build design matrix for W
                # Each rank term r contributes U[:,r] ⊗ V[:,r] scaled by W[c,r]
                # So T[:,:,c] = sum_r W[c,r] * (U[:,r] ⊗ V[:,r])
                
                # Flatten: T_flat[c] = sum_r W[c,r] * (U[:,r] ⊗ V[:,r])_flat
                # This is a linear system in W[c,:] for each c
                
                # Better: iterate over all (a,b,c)
                # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                
                # Solve for W: for each c, we have equations T[a,b,c] = sum_r W[c,r]*U[a,r]*V[b,r]
                # This is linear in W[c,:]
                
                # Build A matrix: A[(a,b), r] = U[a,r]*V[b,r]
                # Then T_flat[c] = A @ W[c,:]
                
                A = np.zeros((n*m*m*p, target_rank))
                idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        for r in range(target_rank):
                            A[idx, r] = U[a, r] * V[b, r]
                        idx += 1
                
                # T_flat[c] = T.reshape(-1, p*n)[:, c]
                T_mat = T.reshape(n*m*m*p, p*n)
                
                # Solve A @ W[c,:] = T_mat[:, c] for each c
                # Use least squares
                try:
                    W_new, _, _, _ = np.linalg.lstsq(A, T_mat, rcond=None)
                    W = W_new
                except:
                    break
                
                # Check convergence
                residual = np.linalg.norm(A @ W - T_mat)
                if residual < 1e-6:
                    # Verify exactness
                    T_recon = np.zeros((n*m, m*p, p*n))
                    for r in range(target_rank):
                        T_recon += np.outer(np.outer(U[:,r], V[:,r]), W[:,r]).reshape(n*m, m*p, p*n)
                    if np.allclose(T_recon, T, atol=1e-8):
                        found = True
                        best_rank = target_rank
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                        break
                
                # Update U and V similarly
                # Solve for U: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # For each a, T[a,:,:] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
                A_u = np.zeros((m*p*p*n, target_rank))
                idx = 0
                for b in range(m*p):
                    for c in range(p*n):
                        for r in range(target_rank):
                            A_u[idx, r] = V[b, r] * W[c, r]
                        idx += 1
                T_u = T.reshape(n*m, m*p*p*n).T  # shape (m*p*p*n, n*m)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(A_u, T_u, rcond=None)
                    U = U_new.T
                except:
                    break
                
                # Solve for V
                A_v = np.zeros((n*m*p*n, target_rank))
                idx = 0
                for a in range(n*m):
                    for c in range(p*n):
                        for r in range(target_rank):
                            A_v[idx, r] = U[a, r] * W[c, r]
                        idx += 1
                T_v = T.reshape(m*p, n*m*p*n).T  # shape (n*m*p*n, m*p)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(A_v, T_v, rcond=None)
                    V = V_new.T
                except:
                    break
                
                if found:
                    break
            
            if found:
                break
        
        if found:
            break
    
    return best_result
