"""
Program for rank-minimal exact tensor decomposition of 3x3 by 3x7 matrix multiplication.

Uses a structured approach: start with schoolbook (rank 63), then attempt to
reduce rank using iterative optimization with random restarts and verification.
"""

import time
import numpy as np
import random

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Returns:
        dict: A dictionary containing 'U', 'V', 'W' matrices.
        U has shape (n*m, rank)
        V has shape (m*p, rank)
        W has shape (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21

    # Build the target tensor T[a,b,c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook decomposition (rank 63)
    rank_schoolbook = n * m * p  # 63
    U_school = np.zeros((nm, rank_schoolbook))
    V_school = np.zeros((mp, rank_schoolbook))
    W_school = np.zeros((pn, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        """Check if the decomposition is valid."""
        rank = U_mat.shape[1]
        # Check all non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_mat[a, :] * V_mat[b, :] * W_mat[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check all zero entries (comprehensive check)
        nonzero_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    nonzero_set.add((i*m+j, j*p+k, k*n+i))
        
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) not in nonzero_set:
                        val = np.sum(U_mat[a, :] * V_mat[b, :] * W_mat[c, :])
                        if abs(val) > tol:
                            return False
        return True

    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = rank_schoolbook

    # Try to find lower-rank decompositions using iterative optimization
    # We'll try ranks from 62 down to 40
    # Use a more efficient optimization approach
    
    def try_rank(target_rank, num_restarts=5, max_iter=1000, lr_init=0.01):
        """Try to find a decomposition of given rank using gradient descent."""
        for attempt in range(num_restarts):
            if time.time() > deadline - 20.0:
                return None, None, None
            
            np.random.seed(42 + attempt * 137)
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.random((pn, target_rank))  # Start with positive values
            W = W * 2 - 1  # Center around 0
            
            lr = lr_init
            prev_error = float('inf')
            stuck_count = 0
            
            for step in range(max_iter):
                if time.time() > deadline - 20.0:
                    break
                
                # Compute the current tensor approximation
                # T_current[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # For efficiency, compute error on a random subset
                
                num_samples = 200
                a_idx = np.random.randint(0, nm, num_samples)
                b_idx = np.random.randint(0, mp, num_samples)
                c_idx = np.random.randint(0, pn, num_samples)
                
                # Compute T_current for these samples
                T_cur = np.zeros(num_samples)
                for s in range(num_samples):
                    T_cur[s] = np.sum(U[a_idx[s], :] * V[b_idx[s], :] * W[c_idx[s], :])
                
                T_target = T[a_idx, b_idx, c_idx]
                
                # Compute error
                error = np.mean((T_target - T_cur)**2)
                
                # Check for convergence
                if error < 1e-10:
                    # Verify the full result
                    if verify(U, V, W):
                        return U, V, W
                    break
                
                # Check for stagnation
                if abs(prev_error - error) < 1e-12:
                    stuck_count += 1
                    if stuck_count > 50:
                        break
                else:
                    stuck_count = 0
                
                prev_error = error
                
                # Compute gradients (approximate using samples)
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for s in range(num_samples):
                    diff = T_target[s] - T_cur[s]
                    for r in range(target_rank):
                        grad_U[a_idx[s], r] += diff * V[b_idx[s], r] * W[c_idx[s], r]
                        grad_V[b_idx[s], r] += diff * U[a_idx[s], r] * W[c_idx[s], r]
                        grad_W[c_idx[s], r] += diff * U[a_idx[s], r] * V[b_idx[s], r]
                
                # Update with adaptive learning rate
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Decay learning rate
                if step % 100 == 0 and step > 0:
                    lr *= 0.5
                
                # Periodic verification
                if step % 200 == 0:
                    if verify(U, V, W):
                        return U, V, W
        
        return None, None, None

    # Try ranks from 62 down to 40
    for target_rank in range(62, 39, -1):
        if time.time() > deadline - 30.0:
            break
        
        U_new, V_new, W_new = try_rank(target_rank, num_restarts=3, max_iter=500)
        
        if U_new is not None:
            best_U = U_new
            best_V = V_new
            best_W = W_new
            best_rank = target_rank
            break  # Found a valid decomposition, stop
    
    # Final check
    if not verify(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U = U_school
        best_V = V_school
        best_W = W_school
        best_rank = rank_schoolbook

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
