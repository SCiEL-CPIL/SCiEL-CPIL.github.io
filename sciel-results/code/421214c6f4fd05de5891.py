"""Program for rank-minimal exact tensor decomposition of 3x5x6 matrix multiplication."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]
TOTAL_TIME_LIMIT = 145.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1.
    """
    rank = U.shape[1]
    if rank == 0:
        return False
        
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute C[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We want C[i,j,k] == 1 for all i,j,k
    C = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
    
    if not np.allclose(C, 1.0, atol=tol, rtol=0):
        return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Returns the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _try_random_decomposition(n, m, p, target_rank, num_trials=100, time_limit=5.0):
    """
    Try to find a decomposition with a specific target rank using random initialization
    and local optimization (gradient descent on the residual).
    """
    start_time = time.time()
    
    # The tensor equation is:
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p equations (90) with 3*rank variables (3*target_rank).
    # We need 3*target_rank >= 90, so target_rank >= 30.
    # But we want to find a solution.
    
    # Strategy:
    # 1. Randomly initialize U, V, W with small values.
    # 2. Use a simple iterative method to solve for the entries.
    #    Since the equations are cubic (product of 3 variables), this is hard.
    #    
    # Alternative Strategy:
    # Fix two matrices (e.g., V and W) randomly, and solve for U.
    # For each (i,j), the equation is:
    # sum_k [ sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ] = 1
    # This is linear in U[i*m+j, :]!
    # Let A_{(i,j), r} = sum_k V[j*p+k, r] * W[k*n+i, r]
    # Then we need: sum_r U[i*m+j, r] * A_{(i,j), r} = 1
    # This is a linear system: A * U_row = 1 (scalar)
    # Wait, A is a vector of length rank. U_row is a vector of length rank.
    # The equation is a dot product. We have one equation for each (i,j) pair.
    # There are n*m such pairs.
    # If rank > n*m, the system is underdetermined. We can pick a particular solution.
    # If rank <= n*m, it's a square or overdetermined system.
    # Actually, we have n*m equations and n*m*rank unknowns (entries of U).
    # So we can always find a solution if the system is consistent.
    # But we also need V and W to be "generic" enough so that the system has a solution.
    # Actually, for ANY V and W, we can solve for U if the matrix of coefficients has full row rank.
    # Let's define M as a matrix of size (n*m) x rank.
    # M_{(i,j), r} = sum_k V[j*p+k, r] * W[k*n+i, r]
    # We need to find U (size n*m x rank) such that U * M^T = ones(n*m)
    # This is U * M^T = 1.
    # This is a linear system for U.
    # If rank >= n*m, we can always find a solution (e.g., least squares, or pseudo-inverse).
    # But we also need the decomposition to be EXACT.
    # If we solve U * M^T = 1 exactly, then the decomposition is exact.
    # So the algorithm is:
    # 1. Pick random V and W.
    # 2. Compute M.
    # 3. Solve for U such that U * M^T = 1.
    # 4. Verify.
    
    # However, we want to minimize rank.
    # The minimum rank is at least max(n*m, m*p, p*n) ? No.
    # The rank of the matrix multiplication tensor is at least the "slice rank" or "subrank".
    # For n x m x p, the rank is at least max(n, m, p) * min(n, m, p) ? No.
    # It is known that rank >= max(n*m, m*p, p*n) is NOT true.
    # Actually, rank >= n*m, m*p, p*n is not true.
    # But rank >= max(n, m, p) is true.
    # Also rank >= (n*m*p) / max(n, m, p) ? No.
    
    # Let's just try ranks from 30 upwards.
    # But solving for U requires inverting or pseudo-inverting a (n*m) x rank matrix.
    # If rank < n*m, the system U * M^T = 1 might not have a solution.
    # If rank >= n*m, it always has a solution.
    # So the minimum rank we can find this way is n*m = 15.
    # But is 15 possible?
    # If rank = 15, M is 15x15. We need det(M) != 0.
    # M_{(i,j), r} = sum_k V[j*p+k, r] * W[k*n+i, r]
    # This looks like a matrix multiplication itself.
    # Let's see if we can make M invertible.
    
    # Let's try rank = n*m = 15.
    # If that fails, try 16, 17, ...
    
    # But wait, the problem asks for "rank-minimal".
    # The schoolbook rank is 90.
    # Can we do better?
    # Yes, likely.
    
    # Let's implement the "Fix V, W, solve for U" approach.
    
    for r in range(target_rank, -1, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try a few random V, W
        for trial in range(num_trials):
            if time.time() - start_time > time_limit:
                break
                
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Construct M
            # M has shape (n*m, r)
            # M[i*m + j, r_idx] = sum_k V[j*p + k, r_idx] * W[k*n + i, r_idx]
            M = np.zeros((n * m, r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        M[i * m + j, :] += V[j * p + k, :] * W[k * n + i, :]
            
            # Solve U * M^T = 1
            # U is (n*m, r)
            # M^T is (r, n*m)
            # Let X = U^T. Then M * X = 1 (vector of ones of length n*m)
            # X is (r, n*m). We need M * X = 1.
            # This is a linear system.
            try:
                # If r >= n*m, we can use least squares or pinv.
                # If r < n*m, we need the system to be consistent.
                X, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n * m), rcond=None)
                
                # Check if the solution is exact
                # M * X should be 1
                if np.allclose(M @ X, np.ones(n * m), atol=1e-9):
                    U = X.T
                    if _verify_decomposition(U, V, W, n, m, p):
                        return U, V, W, r
            except np.linalg.LinAlgError:
                continue
                
    return None

def _try_strassen_like(n, m, p, time_limit=10.0):
    """
    Attempt to find a lower rank decomposition using structured patterns.
    This is a placeholder for more advanced algorithms.
    """
    pass

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for matrix multiplication.
    Dimensions: n=3, m=5, p=6.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook as fallback
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to find a lower rank decomposition
    # Theoretical lower bound: max(n, m, p) = 6
    # But likely higher.
    # Let's try ranks from 15 (n*m) upwards.
    
    # We know that rank must be at least max(n*m, m*p, p*n) is false.
    # But rank must be at least the "tensor rank".
    # For 3x5x6, the rank is at least 15?
    # Actually, the rank of the multiplication tensor for n x m x p is at least max(n*m, m*p, p*n) is NOT a lower bound.
    # The lower bound is max(n, m, p) * min(n, m, p) ? No.
    # A simple lower bound is max(n, m, p).
    # Another is (n*m*p) / max(n, m, p) ? No.
    
    # Let's just iterate from a reasonable lower bound.
    # n*m = 15, m*p = 30, p*n = 18.
    # The rank is at least max(15, 30, 18) = 30?
    # Actually, the rank of the tensor is at least the maximum of the dimensions of the slices?
    # No.
    
    # Let's try ranks from 15 to 90.
    # But we should start from the lower bound.
    # Let's try 15, 16, ..., 90.
    
    # To save time, we can try a few specific ranks.
    # But the "Fix V, W, solve for U" method works for any rank >= n*m = 15.
    # So we can try rank = 15.
    
    # However, the random V, W might not lead to an invertible M.
    # We need M to have full row rank (n*m = 15).
    # If r = 15, M is 15x15. We need det(M) != 0.
    
    # Let's try a few random trials for rank = 15.
    
    for target_rank in range(15, 91):
        if time.time() - start_time > TOTAL_TIME_LIMIT - 5:
            break
            
        result = _try_random_decomposition(n, m, p, target_rank, num_trials=50, time_limit=5.0)
        if result is not None:
            U, V, W, r = result
            if r < best_rank:
                best_rank = r
                best_U, best_V, best_W = U, V, W
                # If we found a solution, we can stop if we hit the theoretical lower bound?
                # The lower bound is not known precisely, but 15 is a candidate.
                # Let's continue to see if we can go lower? No, 15 is n*m.
                # Can rank be < n*m?
                # If rank < n*m, then M is (n*m) x r with r < n*m.
                # Then M * X = 1 might not have a solution.
                # So we can't go below 15 with this method.
                # But maybe a different method can?
                # Unlikely.
                # Let's assume 15 is the minimum for this method.
                # But we should check if we can do better with a different approach.
                # For now, let's return the best found.
                break
        
    # Verify the best result
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fallback to schoolbook
        best_U, best_V, best_W = U, V, W
        best_rank = n * m * p
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
