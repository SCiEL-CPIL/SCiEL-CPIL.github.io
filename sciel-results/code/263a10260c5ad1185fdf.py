"""Matrix multiplication 2x3x4 rank-minimal decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    
    # Build the tensor T of shape (nm, mp, pn)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 iff i==i', j==j', k==k'
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition (rank 24)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Try to find lower rank decompositions
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try ranks from 20 up to 23
    for target_rank in range(20, 24):
        if time.time() > deadline - 10:
            break
        success = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            rng = np.random.RandomState(attempt * 12345 + 67890)
            U = rng.randn(nm, target_rank)
            V = rng.randn(mp, target_rank)
            W = rng.randn(pn, target_rank)
            
            # ALS: alternate fixing two factors and solving for the third
            converged = False
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Fix V, W, solve for U
                # For each row i of U: U[i,:] @ (V[j,:] ⊙ W[k,:])^T = T[i,j,k] for all j,k
                # This is: sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
                # For fixed i: U[i,:] is a vector of length r
                # The equation for all (j,k): sum_r U[i,r] * (V[j,r]*W[k,r]) = T[i,j,k]
                # Let M_i[j,k] = V[j,:] * W[k,:] (element-wise, shape mp x pn)
                # Then U[i,:] @ M_i.T = T[i,:,:].T  (reshaped)
                # Actually: sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
                # Define G_i[j,k,r] = V[j,r] * W[k,r], then sum_r U[i,r] * G_i[j,k,r] = T[i,j,k]
                # This is a linear system: for each i, we have mp*pn equations in target_rank unknowns
                # Reshape: let G_i be shape (mp*pn, target_rank) where G_i[j*p+k, r] = V[j,r]*W[k,r]
                # Then U[i,:] @ G_i.T = T[i,:,:].T.reshape(-1)
                
                # Solve for U
                for i in range(nm):
                    # Build the matrix: for each (j,k) pair, the coefficient vector is V[j,:]*W[k,:]
                    # Shape: (mp*pn, target_rank)
                    G = np.outer(V, W)  # shape (mp, pn, target_rank)? No...
                    # Actually: G[j,k,r] = V[j,r] * W[k,r]
                    # We want: sum_r U[i,r] * G[j,k,r] = T[i,j,k]
                    # Let's build matrix A_i of shape (mp*pn, target_rank)
                    # A_i[j*p+k, r] = V[j,r] * W[k,r]
                    A_i = np.zeros((mp * pn, target_rank))
                    for j in range(mp):
                        for k in range(pn):
                            A_i[j * pn + k, :] = V[j, :] * W[k, :]
                    # b_i = T[i,:,:].reshape(-1)
                    b_i = T[i, :, :].reshape(-1)
                    # Solve A_i @ U[i,:].T = b_i  (least squares)
                    sol, residuals, rank_s, sv = np.linalg.lstsq(A_i, b_i, rcond=None)
                    U[i, :] = sol
                
                # Fix U, W, solve for V
                for j in range(mp):
                    A_j = np.zeros((nm * pn, target_rank))
                    for i in range(nm):
                        for k in range(pn):
                            A_j[i * pn + k, :] = U[i, :] * W[k, :]
                    b_j = T[:, j, :].reshape(-1)
                    sol, residuals, rank_s, sv = np.linalg.lstsq(A_j, b_j, rcond=None)
                    V[j, :] = sol
                
                # Fix U, V, solve for W
                for k in range(pn):
                    A_k = np.zeros((nm * mp, target_rank))
                    for i in range(nm):
                        for j in range(mp):
                            A_k[i * mp + j, :] = U[i, :] * V[j, :]
                    b_k = T[:, :, k].reshape(-1)
                    sol, residuals, rank_s, sv = np.linalg.lstsq(A_k, b_k, rcond=None)
                    W[k, :] = sol
            
            # Check convergence: compute the reconstruction error
            # Compute sum_r U[i,r]*V[j,r]*W[k,r] for all i,j,k and compare to T
            recon = np.zeros((nm, mp, pn))
            for r in range(target_rank):
                recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(nm, mp, pn)
            # Actually: recon[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.max(np.abs(recon - T))
            
            if err < 1e-8:
                success = True
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                break
        
        if success:
            break
    
    # Convert to lists for output
    def to_list(arr):
        return arr.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
