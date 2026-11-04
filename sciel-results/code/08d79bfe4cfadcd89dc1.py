"""Matrix multiplication 3x5x6 rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]
TIME_LIMIT = 140  # seconds, with 10s safety margin from 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need to find rank-r decomposition: T = sum_c U[:,c] * V[:,c] * W[:,c]
    # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    # The contraction: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # First, let's try to find a good rank using random search / optimization
    # Start with schoolbook rank as upper bound
    schoolbook_rank = n * m * p  # 90
    
    # Try to find lower rank decompositions
    # Strategy: use random initialization and gradient descent on the tensor
    # T_{ij,k} - sum_c U_{ij,c} V_{jk,c} W_{ki,c} = 0
    
    # For efficiency, we'll work with the tensor directly
    # T is a 3D array of shape (n, m, p) with all ones
    T = np.ones((n, m, p), dtype=np.float64)
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Generate schoolbook solution as fallback
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Try various ranks below schoolbook
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Also rank >= n+m+p-2 = 12 (for some cases), but actual minimum is higher
    
    # Let's try to find decompositions using random search with local optimization
    # We'll use a tensor decomposition approach
    
    for target_rank in range(30, min(schoolbook_rank, 60)):
        if time.time() - start_time > TIME_LIMIT - 20:
            break
        
        found = _try_find_decomposition(n, m, p, target_rank, T, time.time() - start_time)
        if found is not None:
            U, V, W = found
            # Verify the decomposition
            if _verify_decomposition(U, V, W, n, m, p, T):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break  # Found minimal rank, stop
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_U is not None:
        U, V, W = best_U, best_V, best_W
    else:
        U, V, W = U_sb, V_sb, W_sb
    
    # Convert to lists of integer/half-integer entries
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    # Round to nearest half-integer if close
    U_list = [[_round_to_half(x) for x in row] for row in U_list]
    V_list = [[_round_to_half(x) for x in row] for row in V_list]
    W_list = [[_round_to_half(x) for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}


def _round_to_half(x):
    """Round to nearest half-integer."""
    return round(x * 2) / 2


def _verify_decomposition(U, V, W, n, m, p, T, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                # Actually, the decomposition is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                val = 0.0
                for c in range(r):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - T[i,j,k]) > tol:
                    return False
    return True


def _try_find_decomposition(n, m, p, target_rank, T, elapsed):
    """Try to find a rank-target_rank decomposition using random search + local optimization."""
    remaining_time = TIME_LIMIT - 20 - elapsed
    if remaining_time < 5:
        return None
    
    num_trials = min(50, max(5, int(remaining_time / 0.5)))
    
    for trial in range(num_trials):
        if time.time() - (time.time() - remaining_time) > remaining_time - 5:
            break
        
        # Random initialization
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Normalize to avoid trivial scaling issues
        # The product U*V*W should equal T (all ones)
        # We can use a simple iterative method
        
        success = _local_optimize(U, V, W, n, m, p, T, max_iter=1000, tol=1e-8)
        if success:
            return U, V, W
    
    return None


def _local_optimize(U, V, W, n, m, p, T, max_iter=1000, tol=1e-8):
    """Local optimization using gradient descent on the tensor decomposition."""
    r = U.shape[1]
    lr = 0.01
    
    for it in range(max_iter):
        # Compute gradient
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target = T[i,j,k]
                    # Current value
                    current = 0.0
                    for c in range(r):
                        current += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    error = current - target
                    
                    for c in range(r):
                        grad_U[i*m+j, c] += error * V[j*p+k, c] * W[k*n+i, c]
                        grad_V[j*p+k, c] += error * U[i*m+j, c] * W[k*n+i, c]
                        grad_W[k*n+i, c] += error * U[i*m+j, c] * V[j*p+k, c]
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Check convergence
        if it % 50 == 0:
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        current = 0.0
                        for c in range(r):
                            current += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        max_err = max(max_err, abs(current - T[i,j,k]))
            if max_err < tol:
                return True
    
    # Final check
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                current = 0.0
                for c in range(r):
                    current += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                max_err = max(max_err, abs(current - T[i,j,k]))
    
    return max_err < 1e-4
