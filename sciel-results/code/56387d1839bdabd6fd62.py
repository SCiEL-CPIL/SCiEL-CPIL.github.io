import numpy as np
import time
import sys

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125  # safety margin before 150s kill

    # Target tensor is all ones
    # T[i,j,k] = 1
    
    # Trivial decomposition
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, idx] = 1
                V_trivial[b, idx] = 1
                W_trivial[c, idx] = 1
                idx += 1

    best_rank = trivial_rank
    best_U = U_trivial.copy()
    best_V = V_trivial.copy()
    best_W = W_trivial.copy()

    def check_decomposition(U, V, W, tol=1e-8):
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    def alternating_minimization(target_rank, max_iters=200, tol=1e-8):
        """
        Runs AM to find a decomposition of the given rank.
        Returns (U, V, W) if successful, else None.
        """
        # Random initialization
        # Scale to avoid huge numbers initially
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute Z_r[i,j] = sum_k V_r[j,k] W_r[k,i]
            # V is (m, p, R), W is (p, n, R)
            # We want Z[i, j, r] = sum_k V[j, k, r] W[k, i, r]
            # Note: V indices are (j, k, r), W indices are (k, i, r)
            # einsum: 'jkr,kir->ijr'
            V_3d = V.reshape(m, p, target_rank)
            W_3d = W.reshape(p, n, target_rank)
            Z = np.einsum('jkr,kir->ijr', V_3d, W_3d)
            
            # Update U
            # For each i, j, we have vector Z[i,j,:] and want U[i,j,:] such that dot(U, Z) = 1
            # Min norm solution: U[i,j,:] = Z[i,j,:] / sum(Z[i,j,:]^2)
            Z_sq = np.sum(Z**2, axis=2)
            # Avoid division by zero
            mask = Z_sq > 1e-12
            U_new = np.zeros_like(U)
            U_new_3d = U_new.reshape(n, m, target_rank)
            # Only update where mask is true
            # Z is (n, m, R)
            # We need to divide each slice Z[i,j,:] by Z_sq[i,j]
            # Use broadcasting
            U_new_3d[mask] = Z[mask] / Z_sq[mask, np.newaxis]
            # Where mask is false, we can't satisfy the equation with min norm (0=1 impossible if Z=0)
            # In that case, we might leave it as is or set to 0. 
            # If Z is 0, the term contributes 0 regardless of U. 
            # This implies a structural failure for this rank/initialization.
            
            U = U_new

            # Compute Y_r[j,k] = sum_i U[i,j,r] W[k,i,r]
            # U is (n, m, R), W is (p, n, R)
            # We want Y[j, k, r] = sum_i U[i, j, r] W[k, i, r]
            # einsum: 'ijr,kir->jkr'
            U_3d = U.reshape(n, m, target_rank)
            W_3d = W.reshape(p, n, target_rank)
            Y = np.einsum('ijr,kir->jkr', U_3d, W_3d)
            
            # Update V
            Y_sq = np.sum(Y**2, axis=2)
            mask = Y_sq > 1e-12
            V_new = np.zeros_like(V)
            V_new_3d = V_new.reshape(m, p, target_rank)
            V_new_3d[mask] = Y[mask] / Y_sq[mask, np.newaxis]
            V = V_new

            # Compute X_r[k,i] = sum_j U[i,j,r] V[j,k,r]
            # U is (n, m, R), V is (m, p, R)
            # We want X[k, i, r] = sum_j U[i, j, r] V[j, k, r]
            # einsum: 'ijr,jkr->kir'
            U_3d = U.reshape(n, m, target_rank)
            V_3d = V.reshape(m, p, target_rank)
            X = np.einsum('ijr,jkr->kir', U_3d, V_3d)
            
            # Update W
            X_sq = np.sum(X**2, axis=2)
            mask = X_sq > 1e-12
            W_new = np.zeros_like(W)
            W_new_3d = W_new.reshape(p, n, target_rank)
            W_new_3d[mask] = X[mask] / X_sq[mask, np.newaxis]
            W = W_new

            # Check convergence
            # Compute T
            T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            residual = np.max(np.abs(T - 1.0))
            
            if residual < tol:
                return (U, V, W)
                
            # Optional: damping or restart if stuck? 
            # For now, just continue.
            
        return None

    # Try to find lower rank
    # Start from 119 down to 80
    # Number of trials per rank
    trials_per_rank = 10
    
    for target_rank in range(trivial_rank - 1, 79, -1):
        if time.time() > deadline - 5:
            break
            
        found = False
        for trial in range(trials_per_rank):
            if time.time() > deadline - 2:
                break
                
            result = alternating_minimization(target_rank, max_iters=50)
            if result is not None:
                U, V, W = result
                if check_decomposition(U, V, W):
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break
        
        if found:
            # If we found a lower rank, we can continue to search for even lower
            # or break if we are happy. The goal is minimal rank.
            # We continue the loop to find lower ranks.
            pass

    return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
