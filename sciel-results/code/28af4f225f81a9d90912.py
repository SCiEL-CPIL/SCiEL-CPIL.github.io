"""
Matrix multiplication 4x4 by 4x5: rank-minimal exact tensor decomposition.

We search for a decomposition of the matrix multiplication tensor T[i,j,k] = 1
(where A is 4x4, B is 4x5, C is 4x5) as:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

Strategy:
1. Start with the schoolbook rank-80 solution.
2. Use a randomized ALS (Alternating Least Squares) style search to try to reduce rank.
   For each target rank R from a small set of candidates, we try to find matrices
   U (16 x R), V (20 x R), W (20 x R) with half-integer entries that exactly
   reconstruct the tensor.

Since exact half-integer solutions are hard to find via continuous optimization,
we use a hybrid approach:
- Use continuous ALS to get close to a solution.
- Then round to nearest half-integers and check exactness.
- If exact, we have a valid decomposition.

For 4x4x5, the known minimal rank is not trivially less than 80.
We'll use a practical search with time limits.
"""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

def _build_tensor():
    """Build the matrix multiplication tensor T[i,j,k] for 4x4 * 4x5 -> 4x5."""
    n, m, p = DIMENSIONS
    # T[i,j,k] = 1 for all i,j,k since C[i,k] = sum_j A[i,j] * B[j,k]
    T = np.ones((n, m, p), dtype=np.float64)
    return T

def _tensor_to_vector(T, n, m, p):
    """Flatten tensor to vector in order matching the factorization convention."""
    vec = T.flatten()
    return vec

def _reconstruct_from_factors(U, V, W, n, m, p, R):
    """Reconstruct tensor from factors."""
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
    return T

def _is_exact(T, tol=1e-9):
    """Check if tensor is exactly the multiplication tensor (all entries 1)."""
    n, m, p = DIMENSIONS
    return np.allclose(T, 1.0, atol=tol)

def _round_to_halfint(x):
    """Round to nearest half-integer."""
    return np.round(x * 2) / 2.0

def _verify_solution(U, V, W, n, m, p, R, tol=1e-9):
    """Verify that the factors exactly decompose the tensor."""
    T = _reconstruct_from_factors(U, V, W, n, m, p, R)
    return _is_exact(T, tol)

def construct():
    """Construct a rank-minimal exact tensor decomposition."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 80
    
    # Build the target tensor
    T_target = _build_tensor()
    
    best_rank = schoolbook_rank
    best_solution = None
    
    # Generate schoolbook solution as fallback
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    best_solution = (U_sb, V_sb, W_sb)
    
    # Try to find lower-rank solutions
    # Try ranks from a strategic set
    candidate_ranks = [79, 78, 77, 76, 75, 74, 73, 72, 70, 68, 64]
    
    for R in candidate_ranks:
        if time.time() > deadline - 20:
            break
        
        found = False
        max_trials = 30
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            # ALS iterations
            for it in range(30):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed i,j: T[i,j,k] = sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r])
                # Let A[i,j,k,r] = V[j*p+k,r] * W[k*n+i,r]
                # Then T[i,j,k] = sum_r U[i*m+j,r] * A[i,j,k,r]
                
                for i in range(n):
                    for j in range(m):
                        # Build A[i,j,:,:] which is p x R
                        A_ij = np.zeros((p, R))
                        for k in range(p):
                            A_ij[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        
                        # Solve: A_ij.T @ U[i*m+j,:] = T[i,j,:]
                        lhs = A_ij.T @ A_ij  # R x R
                        rhs = A_ij.T @ T_target[i, j, :]  # R
                        try:
                            U[i * m + j, :] = np.linalg.solve(lhs, rhs)
                        except np.linalg.LinAlgError:
                            U[i * m + j, :] = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
                
                # Update V: for fixed j, solve for V[j*p+k, :] for all k
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed j, unknowns are V[j*p+0,:], ..., V[j*p+p-1,:]
                # For each (i,k): T[i,j,k] = sum_r (U[i*m+j,r] * W[k*n+i,r]) * V[j*p+k,r]
                
                for j in range(m):
                    # Build the linear system for V[j*p+0:p,:]
                    # For each (i,k): sum_r B[i,k,r] * V[j*p+k,r] = T[i,j,k]
                    # where B[i,k,r] = U[i*m+j,r] * W[k*n+i,r]
                    
                    lhs_mat = np.zeros((n * p, p * R))
                    rhs_vec = np.zeros(n * p)
                    idx = 0
                    for i in range(n):
                        for k in range(p):
                            B_ik = U[i * m + j, :] * W[k * n + i, :]
                            lhs_mat[idx, k * R:(k+1) * R] = B_ik
                            rhs_vec[idx] = T_target[i, j, k]
                            idx += 1
                    
                    try:
                        sol = np.linalg.lstsq(lhs_mat, rhs_vec, rcond=None)[0]
                        for k in range(p):
                            V[j * p + k, :] = sol[k * R:(k+1) * R]
                    except:
                        pass
                
                # Update W: for fixed k, solve for W[k*n+i, :] for all i
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed k, unknowns are W[k*n+0,:], ..., W[k*n+n-1,:]
                # For each (i,j): T[i,j,k] = sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r]
                
                for k in range(p):
                    # Build the linear system for W[k*n+0:n,:]
                    # For each (i,j): sum_r C[i,j,r] * W[k*n+i,r] = T[i,j,k]
                    # where C[i,j,r] = U[i*m+j,r] * V[j*p+k,r]
                    
                    lhs_mat = np.zeros((m * n, n * R))
                    rhs_vec = np.zeros(m * n)
                    idx = 0
                    for j in range(m):
                        for i in range(n):
                            C_ij = U[i * m + j, :] * V[j * p + k, :]
                            lhs_mat[idx, i * R:(i+1) * R] = C_ij
                            rhs_vec[idx] = T_target[i, j, k]
                            idx += 1
                    
                    try:
                        sol = np.linalg.lstsq(lhs_mat, rhs_vec, rcond=None)[0]
                        for i in range(n):
                            W[k * n + i, :] = sol[i * R:(i+1) * R]
                    except:
                        pass
            
            # After ALS, round to half-integers
            U_rounded = _round_to_halfint(U)
            V_rounded = _round_to_halfint(V)
            W_rounded = _round_to_halfint(W)
            
            if _verify_solution(U_rounded, V_rounded, W_rounded, n, m, p, R, tol=1e-9):
                best_rank = R
                best_solution = (U_rounded, V_rounded, W_rounded)
                found = True
                break
        
        if found:
            # Try even lower ranks
            continue
    
    # Convert to lists for output
    U_list = [[float(x) for x in row] for row in best_solution[0]]
    V_list = [[float(x) for x in row] for row in best_solution[1]]
    W_list = [[float(x) for x in row] for row in best_solution[2]]
    
    return {"U": U_list, "V": V_list, "W": W_list}
