import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Tries to find a rank lower than the schoolbook rank of 60.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Build the target tensor in flattened form
    # T[i,j,k] = 1 for all valid (i,j,k) - this is the "all ones" in the 
    # appropriate sense for matrix multiplication
    # Actually, the tensor for matrix multiplication C[i,k] = sum_j A[i,j]B[j,k]
    # has the property that the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    # So the tensor T has T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor
    # in the (i,j,k) space, but with the constraint that the same j appears in A and B).
    
    # The decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # Wait, that's not right either. Let me reconsider.
    
    # The matrix multiplication tensor M is in V* ⊗ W* ⊗ Z where:
    # V = K^{n*m} (input A), W = K^{m*p} (input B), Z = K^{n*p} (output C)
    # M(A, B) = A*B
    # In components: M[i,j, k,l] where i,j index A and k,l index B... 
    # No, it's a bilinear map, so M is in V* ⊗ W* ⊗ Z.
    # M_{v,w,z} = coefficient such that z component of M(A,B) = sum_{v,w} M_{v,w,z} A_v B_w
    
    # For matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # So M_{(i,j), (j',k), (i',k')} = 1 if i=i', j=j', k=k' else 0
    # Wait, that's not right. The output index is (i,k) not (i',k').
    # M_{(i,j), (j',k), (i'',k'')} = 1 if i=i'', j=j', k=k'' else 0
    
    # Let me use the flattening from the seed:
    # U has shape (n*m, rank), indexed by (i*m + j, c)
    # V has shape (m*p, rank), indexed by (j*p + k, c)  
    # W has shape (p*n, rank), indexed by (k*n + i, c)
    
    # The condition is: for all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And this should be the only non-zero term... 
    # Actually no, the tensor is defined such that this sum equals the coefficient
    # of A[i,j]*B[j,k] in C[i,k], which is 1.
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 60 equations in rank*3 unknowns (with rank variables).
    # We want the minimum rank such that this system has a solution with rational entries.
    
    # Strategy: Try ranks from some lower bound up to schoolbook_rank
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But actually the lower bound is higher. For matrix multiplication, 
    # the rank is at least n*m*p / max(n,m,p)^... I'm not sure of the exact bound.
    # Let's just try from a reasonable lower bound.
    
    # Actually, let me try a different approach: use the schoolbook as a starting point
    # and try to reduce rank by finding linear dependencies.
    
    # First, let's just return the schoolbook solution as a baseline,
    # but try to improve it if we have time.
    
    def schoolbook_decomposition():
        rank = schoolbook_rank
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
                    
        return U, V, W
    
    # Try to find a lower rank decomposition using numerical optimization
    # We'll try ranks from a lower bound up to schoolbook_rank - 1
    
    def check_decomposition(U, V, W):
        """Check if the decomposition is valid."""
        # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True
    
    def snap_to_rational(arr, max_denom=100):
        """Try to snap values to simple rationals."""
        result = arr.copy()
        for idx in np.ndindex(result.shape):
            val = result[idx]
            # Try small fractions
            found = False
            for denom in range(1, max_denom + 1):
                num = round(val * denom)
                if abs(val - num/denom) < 1e-8:
                    result[idx] = num / denom
                    found = True
                    break
            if not found:
                # Try half-integers and integers
                if abs(val - round(val)) < 1e-8:
                    result[idx] = float(round(val))
                elif abs(val * 2 - round(val * 2)) < 1e-8:
                    result[idx] = float(round(val * 2)) / 2.0
        return result
    
    # Try to find a lower rank using alternating minimization
    def try_rank_reduction(target_rank, max_iter=1000, tol=1e-10):
        """Try to find a decomposition of the given rank."""
        if target_rank >= schoolbook_rank:
            return None
        
        U, V, W = schoolbook_decomposition()
        
        # Truncate to target_rank
        U_t = U[:, :target_rank].copy()
        V_t = V[:, :target_rank].copy()
        W_t = W[:, :target_rank].copy()
        
        # Alternating least squares
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Update U: for each (i,j), solve for U[i*m+j, :]
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # This is: U[i*m+j, :] @ (V[j*p+k, :] * W[k*n+i, :]) = 1
            # For fixed (j,k), we have a linear system in U[i*m+j, :]
            # But it's coupled across k for fixed i,j
            
            # Let's do a simpler approach: gradient descent
            # Compute the loss
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_t[i*m+j, :] * V_t[j*p+k, :] * W_t[k*n+i, :])
                        loss += (val - 1.0) ** 2
            
            if loss < tol:
                break
            
            # Gradient with respect to U
            grad_U = np.zeros_like(U_t)
            grad_V = np.zeros_like(V_t)
            grad_W = np.zeros_like(W_t)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_t[i*m+j, :] * V_t[j*p+k, :] * W_t[k*n+i, :])
                        err = val - 1.0
                        grad_U[i*m+j, :] += 2 * err * V_t[j*p+k, :] * W_t[k*n+i, :]
                        grad_V[j*p+k, :] += 2 * err * U_t[i*m+j, :] * W_t[k*n+i, :]
                        grad_W[k*n+i, :] += 2 * err * U_t[i*m+j, :] * V_t[j*p+k, :]
            
            # Simple gradient step
            lr = 0.01
            U_t -= lr * grad_U
            V_t -= lr * grad_V
            W_t -= lr * grad_W
        
        # Check if we found a valid solution
        if check_decomposition(U_t, V_t, W_t):
            # Try to snap to rationals
            U_r = snap_to_rational(U_t)
            V_r = snap_to_rational(V_t)
            W_r = snap_to_rational(W_t)
            if check_decomposition(U_r, V_r, W_r):
                return U_r, V_r, W_r
            else:
                return U_t, V_t, W_t
        
        return None
    
    # Try ranks from lower bound up to schoolbook_rank
    # Lower bound estimate: for n x m * m x p, rank >= n*m*p / max(n,m,p)
    # This is a rough bound. Let's try from max(n*m, m*p, p*n) = 30
    lower_bound = max(n * m, m * p, p * n)
    
    best_result = None
    for target_rank in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 10.0:
            break
        
        result = try_rank_reduction(target_rank, max_iter=500)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fall back to schoolbook
    U, V, W = schoolbook_decomposition()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
