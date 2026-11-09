"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
from itertools import combinations

def construct():
    """
    Search for a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    Uses linear algebra to attempt rank reduction from schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor as a 3D array
    # T[i, j, k] = delta_{j,k}
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            T[i, j, j] = 1.0
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Attempt rank reduction using linear algebra
    # The key insight: if we can find linear dependencies among the columns,
    # we can reduce the rank.
    
    def try_rank_reduction(U, V, W, target_rank, max_time=5.0):
        """
        Try to reduce the rank of the decomposition to target_rank.
        Uses the fact that the tensor decomposition must satisfy:
        sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = T[i,j,k]
        """
        elapsed = time.time() - start_time
        if elapsed > deadline - max_time:
            return None
        
        # For a given rank r, we need to find U, V, W such that
        # the tensor equation holds. This is a system of equations.
        
        # Flatten the tensor equation
        # For each (i,j,k): sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = T[i,j,k]
        
        # This is nonlinear. Instead, we can try a different approach:
        # Fix two matrices and solve for the third, or use iterative methods.
        
        # Alternative: Try to find if there's a decomposition with rank < current
        # by checking linear dependencies.
        
        # For now, let's try a simple approach: check if we can remove columns
        # that are linear combinations of others.
        
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        
        current_rank = U_curr.shape[1]
        
        # Try to identify redundant columns
        # A column t is redundant if there exist coefficients such that
        # the contribution of column t can be replicated by other columns.
        
        # This is complex. Let's try a simpler heuristic:
        # Check if any two columns are identical or proportional.
        
        for t1 in range(current_rank):
            for t2 in range(t1 + 1, current_rank):
                # Check if columns t1 and t2 are proportional
                col1_U = U_curr[:, t1]
                col2_U = U_curr[:, t2]
                col1_V = V_curr[:, t1]
                col2_V = V_curr[:, t2]
                col1_W = W_curr[:, t1]
                col2_W = W_curr[:, t2]
                
                # Check if col1 is a scalar multiple of col2
                if np.allclose(col1_U, 0) and np.allclose(col2_U, 0):
                    continue
                if np.allclose(col1_V, 0) and np.allclose(col2_V, 0):
                    continue
                if np.allclose(col1_W, 0) and np.allclose(col2_W, 0):
                    continue
                
                # Find ratio from non-zero elements
                ratios = []
                for a, b in zip(col1_U, col2_U):
                    if abs(b) > 1e-10:
                        ratios.append(a / b)
                for a, b in zip(col1_V, col2_V):
                    if abs(b) > 1e-10:
                        ratios.append(a / b)
                for a, b in zip(col1_W, col2_W):
                    if abs(b) > 1e-10:
                        ratios.append(a / b)
                
                if ratios:
                    ratio = np.mean(ratios)
                    if np.allclose(col1_U, ratio * col2_U, atol=1e-10) and \
                       np.allclose(col1_V, ratio * col2_V, atol=1e-10) and \
                       np.allclose(col1_W, ratio * col2_W, atol=1e-10):
                        # Column t1 is proportional to t2, can remove t1
                        U_curr = np.delete(U_curr, t1, axis=1)
                        V_curr = np.delete(V_curr, t1, axis=1)
                        W_curr = np.delete(W_curr, t1, axis=1)
                        break
            
            if time.time() - start_time > deadline - max_time:
                break
        
        # Verify the decomposition
        U_new = U_curr
        V_new = V_curr
        W_new = W_curr
        new_rank = U_new.shape[1]
        
        # Check if the decomposition is still valid
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_new[i*m+j, :] * V_new[j*p+k, :] * W_new[k*n+i, :])
                    expected = 1.0 if j == k else 0.0
                    if abs(val - expected) > 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid and new_rank < current_rank:
            return U_new, V_new, W_new
        
        return None
    
    # Try to reduce rank iteratively
    U_curr = U_school.copy()
    V_curr = V_school.copy()
    W_curr = W_school.copy()
    
    improved = True
    while improved and (time.time() - start_time) < deadline - 10:
        improved = False
        result = try_rank_reduction(U_curr, V_curr, W_curr, U_curr.shape[1] - 1, max_time=20.0)
        if result is not None:
            U_curr, V_curr, W_curr = result
            improved = True
    
    if U_curr.shape[1] < best_rank:
        best_rank = U_curr.shape[1]
        best_U = U_curr
        best_V = V_curr
        best_W = W_curr
    
    # Convert to lists of lists with integer/half-integer entries
    def to_list(mat):
        # Round to nearest half-integer
        rounded = np.round(mat * 2) / 2.0
        return [[float(rounded[i][j]) for j in range(rounded.shape[1])] for i in range(rounded.shape[0])]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
