"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import random

def construct():
    """
    Find a low-rank exact tensor decomposition for 2x2 * 2x8 matrix multiplication.
    Uses structured random initialization with iterative refinement.
    """
    start_time = time.time()
    deadline = 145.0  # 5 second safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # The tensor T[i,j,k] for matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In the trace convention: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{j,k}
    # Wait, that's not right. Let me reconsider.
    # 
    # Actually for the multiplication tensor:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the "target" is: for each (i,j,k), the coefficient should be 1 if this term
    # contributes to C[i,k], which it always does. So T[i,j,k] = 1 for all valid (i,j,k)?
    # No wait. The tensor decomposition satisfies:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] is the structure tensor of matrix multiplication.
    # For matrix mult, T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
    # contribute to (i,k) entry of C. This is always 1 for valid indices.
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    def compute_error(U, V, W, n, m, p):
        """Compute the Frobenius norm of the error between the decomposition and target tensor."""
        rank = U.shape[1]
        # Build the decomposed tensor
        # T_decomp[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Target: T_target[i,j,k] = 1 for all i,j,k
        error_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    error_sum += (total - 1.0) ** 2
        return error_sum
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        """Verify that the decomposition is exact."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    # Start with a known valid decomposition (schoolbook, rank 32)
    def schoolbook_decomposition(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    U32, V32, W32 = schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U32.copy(), V32.copy(), W32.copy()
    best_rank = 32
    
    # Try to find lower rank decompositions using iterative refinement
    # Strategy: for each target rank r, try multiple random initializations
    # and use a simple iterative method to reduce the error
    
    for target_rank in range(20, 33):
        if time.time() - start_time > deadline - 10:
            break
        
        max_attempts = 20 if target_rank >= 24 else 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Iterative refinement: for each tensor factor, fix the other two
            # and solve for the best update
            converged = False
            for iteration in range(200):
                if time.time() - start_time > deadline - 3:
                    break
                
                old_error = compute_error(U, V, W, n, m, p)
                
                # Update U: for each (i,j), find best U[i*m+j, :] given V and W
                for i in range(n):
                    for j in range(m):
                        # We want to minimize sum_k (sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1)^2
                        # This is a linear system in U[i*m+j, :]
                        # Let a_r = V[j*p+k, r] * W[k*n+i, r] summed over k... 
                        # Actually, let's define:
                        # For fixed (i,j), the contribution to error from varying U[i*m+j, :] is:
                        # sum_k (sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] - 1)^2
                        # Let c_k = 1 (target)
                        # Let b_{k,r} = V[j*p+k, r] * W[k*n+i, r]
                        # We want to minimize sum_k (sum_r U[i*m+j, r] * b_{k,r} - c_k)^2
                        # This is a least squares problem: B @ u = c where B[k,r] = b_{k,r}
                        B_mat = np.zeros((p, target_rank))
                        for k in range(p):
                            for r in range(target_rank):
                                B_mat[k, r] = V[j * p + k, r] * W[k * n + i, r]
                        c_vec = np.ones(p)
                        # Solve B_mat @ u = c_vec in least squares sense
                        try:
                            u_opt, _, _, _ = np.linalg.lstsq(B_mat, c_vec, rcond=None)
                            U[i * m + j, :] = u_opt
                        except:
                            pass
                
                # Update V: for each (j,k), find best V[j*p+k, :] given U and W
                for j in range(m):
                    for k in range(p):
                        B_mat = np.zeros((n, target_rank))
                        for i in range(n):
                            for r in range(target_rank):
                                B_mat[i, r] = U[i * m + j, r] * W[k * n + i, r]
                        c_vec = np.ones(n)
                        try:
                            v_opt, _, _, _ = np.linalg.lstsq(B_mat, c_vec, rcond=None)
                            V[j * p + k, :] = v_opt
                        except:
                            pass
                
                # Update W: for each (k,i), find best W[k*n+i, :] given U and V
                for k in range(p):
                    for i in range(n):
                        B_mat = np.zeros((m, target_rank))
                        for j in range(m):
                            for r in range(target_rank):
                                B_mat[j, r] = U[i * m + j, r] * V[j * p + k, r]
                        c_vec = np.ones(m)
                        try:
                            w_opt, _, _, _ = np.linalg.lstsq(B_mat, c_vec, rcond=None)
                            W[k * n + i, :] = w_opt
                        except:
                            pass
                
                new_error = compute_error(U, V, W, n, m, p)
                if abs(old_error - new_error) < 1e-12:
                    converged = True
                    break
                
                if new_error > old_error * 1.1:
                    # Diverging, break
                    break
            
            # Check if we found a valid decomposition
            if verify_decomposition(U, V, W, n, m, p, tol=1e-4):
                # Try to snap to exact values
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
                found = True
                break
        
        if found:
            # Try even lower ranks
            continue
    
    # Round to nearest half-integers if close
    def round_to_halfint(arr, tol=1e-3):
        rounded = np.round(arr * 2) / 2
        if np.allclose(arr, rounded, atol=tol):
            return rounded
        return arr
    
    best_U = round_to_halfint(best_U)
    best_V = round_to_halfint(best_V)
    best_W = round_to_halfint(best_W)
    
    # Final verification
    if not verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-3):
        # Fallback to schoolbook
        U32, V32, W32 = schoolbook_decomposition(n, m, p)
        return {"U": U32.tolist(), "V": V32.tolist(), "W": W32.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
