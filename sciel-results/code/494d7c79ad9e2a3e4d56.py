"""Rewritten program for 3x3x5 matrix multiplication rank minimization."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

# Known lower bound for 3x3x5 matrix multiplication is 20 (over reals)
# Upper bound from schoolbook is 45
# We'll try to find a decomposition of rank < 45

def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for 3x3x5 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Since T is all-ones, we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    
    # Strategy: Try small ranks with random initialization and ALS
    # First, let's try to see if we can do better than 45
    
    best_result = None
    best_rank = n * m * p  # Start with schoolbook
    
    # Try ranks from low to high
    # Known results suggest rank around 20-25 might be possible
    for target_rank in range(20, min(46, n*m*p + 1)):
        if time.time() > deadline - 10:
            break
            
        success = _try_rank(n, m, p, target_rank, deadline)
        if success is not None:
            best_result = success
            best_rank = target_rank
            break  # Found a valid decomposition at this rank
    
    if best_result is None:
        # Fallback to schoolbook
        best_result = _schoolbook(n, m, p)
        best_rank = n * m * p
    
    return best_result


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _try_rank(n, m, p, rank, deadline):
    """
    Try to find a rank-`rank` decomposition using randomized ALS.
    Returns dict or None if failed.
    """
    np.random.seed(42)  # Reproducibility
    
    # The tensor we want to decompose: T[i,j,k] = 1 for all valid i,j,k
    # In flattened form: T_flat[i*m+j, j*p+k, k*n+i] = 1
    
    # We'll use the fact that the target tensor is all-ones
    # Let's parameterize and solve
    
    max_iter = 50
    tol = 1e-8
    
    for attempt in range(3):
        if time.time() > deadline - 5:
            return None
        
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Scale to avoid numerical issues
        U /= np.linalg.norm(U) * np.sqrt(rank)
        V /= np.linalg.norm(V) * np.sqrt(rank)
        W /= np.linalg.norm(W) * np.sqrt(rank)
        
        converged = False
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # This is a system of equations. Let's use a different approach.
            
            # Actually, let's use the structure: the target is all-ones tensor.
            # We can think of this as: find U,V,W such that the tensor 
            # T_c = U_c ⊗ V_c ⊗ W_c sums to the all-ones tensor.
            
            # Alternative: use the constraint that for each (i,j,k):
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            # This is a bilinear system. Let's try a different parameterization.
            
            # Let's use the approach of fixing two factors and solving for the third.
            
            # Fix V and W, solve for U:
            # For each (i,j): sum_c U[i*m+j,c] * (sum_k' V[j*p+k',c] * W[k'*n+i,c]) = 1
            # Wait, this doesn't work directly because the sum is over k.
            
            # Let me reconsider. The constraint is:
            # For all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            # This means the tensor formed by U,V,W must equal the all-ones tensor.
            
            # Let's use a tensor completion approach.
            # Create the full tensor T of shape (n*m, m*p, p*n)
            # T[i*m+j, j*p+k, k*n+i] should equal 1
            
            # Actually, let's just check if our current U,V,W gives a good approximation
            # and use gradient descent or ALS on the residuals.
            
            # Compute current tensor approximation
            # T_approx[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            
            # For efficiency, let's compute residuals and do simple updates.
            
            # Let's use a simpler approach: direct check with random restarts
            # and see if we can find exact solutions.
            
            # Compute the maximum error
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        max_err = max(max_err, abs(val - 1.0))
            
            if max_err < tol:
                converged = True
                break
            
            # Simple gradient step
            lr = 0.01
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Residual
                        resid = 0.0
                        for c in range(rank):
                            resid += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        resid -= 1.0  # Target is 1
                        
                        # Gradient for U[i*m+j, c]
                        for c in range(rank):
                            grad_U = resid * V[j*p+k, c] * W[k*n+i, c]
                            U[i*m+j, c] -= lr * grad_U
                        
                        # Gradient for V[j*p+k, c]
                        for c in range(rank):
                            grad_V = resid * U[i*m+j, c] * W[k*n+i, c]
                            V[j*p+k, c] -= lr * grad_V
                        
                        # Gradient for W[k*n+i, c]
                        for c in range(rank):
                            grad_W = resid * U[i*m+j, c] * V[j*p+k, c]
                            W[k*n+i, c] -= lr * grad_W
        
        if converged:
            # Verify and return
            result = _verify_and_convert(U, V, W, n, m, p, rank)
            if result is not None:
                return result
    
    return None


def _verify_and_convert(U, V, W, n, m, p, rank):
    """Verify the decomposition is exact and convert to list format."""
    tol = 1e-6
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return None
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
