"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (rank n*m*p = 75)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = rank_schoolbook

    # Target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We can verify a decomposition by checking if sum_r U[:,r] V[:,r] W[:,r] matches T
    
    def verify(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Compute T_reconstructed
        # T_reconstructed[i, j, k] = sum_r U[i,r] V[j,r] W[k,r]
        # Note: indices in U, V, W correspond to flattened indices
        # i in U is i*m+j? No, U is indexed by (i,j) flattened to i*m+j.
        # V is indexed by (j,k) flattened to j*p+k.
        # W is indexed by (k,i) flattened to k*n+i.
        
        # Let's check specific entries
        # For all i in range(n), j in range(m), k in range(p):
        # idx_u = i*m + j
        # idx_v = j*p + k
        # idx_w = k*n + i
        # val = dot(U[idx_u, :], V[idx_v, :], W[idx_w, :])? No.
        # val = sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
        # This val must be 1.0
        
        # For all other combinations, it must be 0.0.
        # This is expensive to check all 3*5*5*... entries.
        # But we can just check the 75 non-zero entries and a sample of zero entries.
        # Actually, if the tensor is sparse, we can just reconstruct the full tensor and compare.
        # Size is 15 x 25 x 15 = 5625. Very small.
        
        R = U.shape[1]
        # Efficient computation:
        # T_recon = np.einsum('ir,jr,kr->ijk', U, V, W) 
        # But U, V, W are 2D.
        # Let's reshape to 3D?
        # U is (15, R). V is (25, R). W is (15, R).
        # We need T_recon[i, j, k] where i in 0..14, j in 0..24, k in 0..14.
        # The mapping is:
        # i_flat = i*m + j  (i in 0..2, j in 0..4) -> 0..14
        # j_flat = j*p + k  (j in 0..4, k in 0..4) -> 0..24
        # k_flat = k*n + i  (k in 0..4, i in 0..2) -> 0..14
        
        # It's easier to just iterate over the 75 non-zero positions and check if they are 1.
        # And check a few zero positions.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.dot(U[idx_u, :], V[idx_v, :]) # This is wrong.
                    # val = sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    val = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check some zero entries
        # Example: i=0, j=0, k=0 is 1.
        # i=0, j=0, k=1 is 0?
        # idx_u = 0, idx_v = 1, idx_w = 1*3 + 0 = 3.
        # Let's check all entries to be safe, 5625 is small.
        
        # Reconstruct full tensor
        # T_recon = np.zeros((15, 25, 15))
        # for r in range(R):
        #    T_recon += np.outer(np.outer(U[:,r], V[:,r]), W[:,r]) # This shape is (15, 25, 15)
        #    # U[:,r] is (15,), V[:,r] is (25,), W[:,r] is (15,)
        #    # np.outer(np.outer(U[:,r], V[:,r]), W[:,r]) results in (15, 25, 15)
        #    # Let's verify:
        #    # A = U[:,r] (15)
        #    # B = V[:,r] (25)
        #    # C = W[:,r] (15)
        #    # AB = A[:,None] * B[None,:] -> (15, 25)
        #    # ABC = AB[:,:,None] * C[None,None,:] -> (15, 25, 15)
        
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(R):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
        # Target T
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        
        if not np.allclose(T_recon, T_target, atol=1e-9):
            return False
            
        return True

    # Search for lower rank
    # We try ranks 74, 73, ...
    # Using random search with rounding to half-integers is a good heuristic.
    
    for rank in range(rank_schoolbook - 1, 69, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations
        found = False
        for attempt in range(20): # Limited attempts
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            # Entries should be small integers or half-integers
            # Let's try values from {-1, -0.5, 0, 0.5, 1}
            U_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
            V_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
            W_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
            
            # Optimize? Or just check?
            # Random check is very unlikely to hit the solution.
            # We need a local search.
            
            # Let's use a simple gradient descent on the continuous relaxation, then round.
            # Objective: || T_target - sum_r U_r V_r W_r ||^2
            # This is non-convex.
            
            # Given the time constraints, a full gradient descent might be too slow.
            # Let's try a few steps of ALS (Alternating Least Squares)
            # Fix V, W, solve for U.
            # U = argmin_U || T - U (V^T W^T) ||_F
            # This is a linear least squares problem for each column of U?
            # No, T is 3D.
            # T_ijk = sum_r U_ir V_jr W_kr
            # For fixed V, W, the problem for U is:
            # min_U sum_{i,j,k} (T_ijk - sum_r U_ir V_jr W_kr)^2
            # This is equivalent to min_U || T - U \otimes (V \otimes W) ||
            # We can reshape T to (n*m, m*p*p*n)? No.
            # Let's reshape T to (n*m, m*p, p*n).
            # For each i, we have a matrix T_i of size (m*p, p*n).
            # T_i[j, k] = sum_r U_ir V_jr W_kr = U_ir (V_r \otimes W_r)^T ? No.
            # Let X_r = V_r \otimes W_r (size m*p*p*n? No, V_r is vector, W_r is vector).
            # T_i[j, k] = sum_r U_ir V_jr W_kr.
            # This is sum_r U_ir (V_r[j] W_r[k]).
            # So T_i = sum_r U_ir (V_r W_r^T).
            # So T_i is a linear combination of rank-1 matrices M_r = V_r W_r^T.
            # We want to find coefficients U_ir such that T_i = sum_r U_ir M_r.
            # This is a least squares problem: T_i (flattened) = A_i U_i
            # where A_i is a matrix whose columns are the flattened M_r.
            
            # Let's implement one step of ALS.
            
            U_curr = U_rand
            V_curr = V_rand
            W_curr = W_rand
            
            for step in range(10): # Few steps
                if time.time() > deadline - 10:
                    break
                    
                # Update U
                # For each i in 0..n*m-1:
                # T_i is (m*p, p*n)
                # M_r is (m*p, p*n) formed by V_curr[:,r] and W_curr[:,r]
                # A is (m*p*p*n, rank)
                # T_i_flat is (m*p*p*n, 1)
                # Solve A u_i = T_i_flat
                
                # Construct A
                # A[:, r] = flatten(V_curr[:, r] * W_curr[:, r]^T)
                # V_curr[:, r] is (m*p, 1), W_curr[:, r] is (p*n, 1)
                # Outer product is (m*p, p*n)
                # Flatten to (m*p*p*n, 1)
                
                A = np.zeros((m*p*p*n, rank))
                for r in range(rank):
                    M_r = np.outer(V_curr[:, r], W_curr[:, r])
                    A[:, r] = M_r.flatten()
                
                # Solve for each i
                for i in range(n*m):
                    T_i = T_target[i, :, :]
                    T_i_flat = T_i.flatten()
                    # Solve A u = T_i_flat
                    # Least squares
                    u_i, residuals, rank_ls, sv = np.linalg.lstsq(A, T_i_flat, rcond=None)
                    U_curr[i, :] = u_i
                    
                # Update V
                # Similar logic
                # T_j is (n*m, p*n)? No.
                # T_ijk = sum_r U_ir V_jr W_kr
                # For fixed U, W, solve for V_j.
                # T_j[i, k] = sum_r U_ir V_jr W_kr
                # T_j = sum_r V_jr (U_r W_r^T)
                # M'_r = U_r W_r^T (size n*m, p*n)
                # A' is (n*m*p*n, rank)
                
                A_prime = np.zeros((n*m*p*n, rank))
                for r in range(rank):
                    M_prime_r = np.outer(U_curr[:, r], W_curr[:, r])
                    A_prime[:, r] = M_prime_r.flatten()
                    
                for j in range(m*p):
                    T_j = T_target[:, j, :] # Shape (n*m, p*n)
                    T_j_flat = T_j.flatten()
                    v_j, residuals, rank_ls, sv = np.linalg.lstsq(A_prime, T_j_flat, rcond=None)
                    V_curr[j, :] = v_j
                    
                # Update W
                # T_k is (n*m, m*p)
                # T_k[i, j] = sum_r U_ir V_jr W_kr
                # T_k = sum_r W_kr (U_r V_r^T)
                # M''_r = U_r V_r^T (size n*m, m*p)
                # A'' is (n*m*m*p, rank)
                
                A_double_prime = np.zeros((n*m*m*p, rank))
                for r in range(rank):
                    M_double_prime_r = np.outer(U_curr[:, r], V_curr[:, r])
                    A_double_prime[:, r] = M_double_prime_r.flatten()
                    
                for k in range(p*n):
                    T_k = T_target[:, :, k] # Shape (n*m, m*p)
                    T_k_flat = T_k.flatten()
                    w_k, residuals, rank_ls, sv = np.linalg.lstsq(A_double_prime, T_k_flat, rcond=None)
                    W_curr[k, :] = w_k
            
            # Round to half-integers
            U_rounded = np.round(U_curr * 2) / 2
            V_rounded = np.round(V_curr * 2) / 2
            W_rounded = np.round(W_curr * 2) / 2
            
            if verify(U_rounded, V_rounded, W_rounded):
                best_U = U_rounded
                best_V = V_rounded
                best_W = W_rounded
                best_rank = rank
                found = True
                break
                
        if found:
            break
            
    # Return the best found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
