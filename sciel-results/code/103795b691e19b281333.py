"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 9, 9, 9
    
    # Build the matrix multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # In the factorization: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note: W uses trace convention W[k*n+i, r]
    
    # The tensor T is indexed by (i, j, k) where:
    # U has shape (n*m, rank) = (9, rank), indexed by (i,j) -> i*m+j
    # V has shape (m*p, rank) = (9, rank), indexed by (j,k) -> j*p+k
    # W has shape (p*n, rank) = (9, rank), indexed by (k,i) -> k*n+i
    
    # For 3x3 matrix mult, the tensor T[i,j,k] = 1 for all (i,j,k) in [0,3)^3
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Known results: The minimum rank for 3x3 matrix multiplication is 23 (Strassen-like)
    # Actually, for 3x3, the best known is 23 (Landsberg, 1988)
    # But let's try to find a good decomposition.
    
    # First, let's use the schoolbook method as a fallback (rank 27)
    def schoolbook():
        rank = 27
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # Try to find a lower rank decomposition using numerical optimization
    # We'll use a greedy approach with random initialization and local search
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is exact (within tolerance)."""
        # Compute the reconstructed tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def to_rational(U, V, W, max_denom=2):
        """Convert to integer/half-integer if possible."""
        U_r = np.zeros_like(U)
        V_r = np.zeros_like(V)
        W_r = np.zeros_like(W)
        
        for mat in [U_r, V_r, W_r]:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat[i, j]
                    # Try to find a rational with small denominator
                    found = False
                    for denom in [1, 2]:
                        num = round(val * denom)
                        if abs(val - num/denom) < 1e-9:
                            mat[i, j] = num/denom
                            found = True
                            break
                    if not found:
                        # Keep as is, but this means it's not integer/half-integer
                        pass
        return U_r, V_r, W_r
    
    # Try different ranks from low to high
    best_result = None
    best_rank = 27
    
    # Start with known good ranks and try to improve
    # The theoretical minimum for 3x3 is 19 (lower bound), best known is 23
    
    for target_rank in [23, 24, 25, 26, 27]:
        if time.time() > deadline - 30:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Try to find a decomposition of this rank
        success = False
        attempts = 0
        max_attempts = 500
        
        while attempts < max_attempts and time.time() < deadline - 30:
            attempts += 1
            
            # Random initialization
            U = np.random.randn(9, target_rank) * 0.5
            V = np.random.randn(9, target_rank) * 0.5
            W = np.random.randn(9, target_rank) * 0.5
            
            # Use gradient-based optimization to minimize the error
            # Error = sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
            
            def compute_error(U, V, W):
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err += (val - 1.0) ** 2
                return err
            
            # Simple coordinate descent / local search
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 30:
                    break
                
                err = compute_error(U, V, W)
                if err < 1e-12:
                    success = True
                    break
                
                # Update U
                for i in range(9):
                    for r in range(target_rank):
                        grad = 0.0
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for rr in range(target_rank):
                                    val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                                grad += 2 * (val - 1.0) * V[j*p+k, r] * W[k*n+i, r]
                        U[i, r] -= lr * grad / target_rank
                
                # Update V
                for jk in range(9):
                    j = jk // p
                    k = jk % p
                    for r in range(target_rank):
                        grad = 0.0
                        for i in range(n):
                            val = 0.0
                            for rr in range(target_rank):
                                val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                            grad += 2 * (val - 1.0) * U[i*m+j, r] * W[k*n+i, r]
                        V[jk, r] -= lr * grad / target_rank
                
                # Update W
                for ki in range(9):
                    k = ki // n
                    i = ki % n
                    for r in range(target_rank):
                        grad = 0.0
                        for j in range(m):
                            val = 0.0
                            for rr in range(target_rank):
                                val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                            grad += 2 * (val - 1.0) * U[i*m+j, r] * V[j*p+k, r]
                        W[ki, r] -= lr * grad / target_rank
                
                # Adaptive learning rate
                if step % 50 == 49:
                    lr *= 0.95
            
            if success:
                break
        
        if success:
            # Convert to rational if possible
            U_r, V_r, W_r = to_rational(U, V, W)
            if verify(U_r, V_r, W_r, tol=1e-6):
                best_rank = target_rank
                best_result = (U_r, V_r, W_r)
                break
    
    if best_result is None:
        U, V, W = schoolbook()
    else:
        U, V, W = best_result
    
    # Convert to list format
    U_list = [[float(U[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i, j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i, j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
