"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U @ V @ W^T reconstructs the matmul tensor."""
    rank = U.shape[1]
    # Build the matmul tensor T[i*m+j, k*n+i, r] where r = k*n + i is W index
    # Actually, the convention: W[k*n+i, r] means W row index = k*n+i
    # The tensor for (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # T[i*m+j, k*n+i, col] should equal delta_{j, j} for the product... 
    # Let me think: we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i} * delta_{k,k} = 1
    # Wait, no. The matmul tensor is T_{i,j,k} = A_{i,j} B_{j,k} -> C_{i,k}
    # So the tensor has indices (i, j, k) and T[i,j,k] = 1 if we're looking at the identity-like tensor
    # Actually the "matrix multiplication tensor" has T[i,j,k] = 1 for all valid i,j,k
    # No wait - it's the structural tensor: T[i*m+j, k*n+i, r] such that
    # C[i,k] = sum_j sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    # For this to equal sum_j A[i,j]B[j,k], we need:
    # sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = delta_{j,j} = 1 (for the coefficient of A[i,j]B[j,k])
    # But also = 0 when the indices don't match in the right way.
    # Actually the standard formulation: T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # And we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # Wait that's not right either. Let me reconsider.
    
    # The matrix multiplication tensor: for multiplying (n x m) by (m x p) to get (n x p)
    # C_{i,k} = sum_{j=0}^{m-1} A_{i,j} B_{j,k}
    # The tensor T has T[i,j,k] = 1 for all valid indices
    # We decompose: T[i,j,k] = sum_{r=1}^{rank} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Check a sample of indices for speed
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check all entries (small enough for 3,4,8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False, abs(val - 1.0)
    return True, 0.0

def _random_init(n, m, p, rank, seed=None):
    """Generate random factor matrices with structured initialization."""
    if seed is not None:
        np.random.seed(seed)
    
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Scale to help convergence
    scale = (rank / (n*m*p)) ** (1/3)
    U *= scale
    V *= scale
    W *= scale
    
    return U, V, W

def _refine(U, V, W, n, m, p, max_iter=200, tol=1e-10):
    """Iterative refinement: alternate solving for each factor given the other two."""
    U = U.copy().astype(np.float64)
    V = V.copy().astype(np.float64)
    W = W.copy().astype(np.float64)
    rank = U.shape[1]
    
    for it in range(max_iter):
        # Solve for U given V, W: for each (i,j), U[i*m+j, :] should satisfy
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
        # This is a system: for fixed (i,j), we have p equations in rank unknowns
        # Let M_{k,r} = V[j*p+k, r] * W[k*n+i, r]
        # Then sum_r U[i*m+j, r] * M_{k,r} = 1 for all k
        # If rank >= p, this is underdetermined; use least squares or pseudo-inverse
        
        # Similarly for V and W
        
        # Actually, let's use a simpler gradient-based approach
        # Compute residual gradient
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        # For each column r, the contribution to T[i,j,k] is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # We want sum_r of that = 1
        
        # Gradient descent with adaptive step
        lr = 0.1
        for inner in range(5):
            # Compute current tensor approximation
            # grad_U[i*m+j, r] = sum_k (T_approx[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r]
            # But T_approx[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            
            # Compute T_approx
            T_approx = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_approx[i,j,k] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
            
            residual = T_approx - 1.0
            
            # Gradient for U
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res = residual[i,j,k]
                        for r in range(rank):
                            grad_U[i*m+j, r] += res * V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += res * U[i*m+j, r] * W[k*n+i, r]
                            grad_W[k*n+i, r] += res * U[i*m+j, r] * V[j*p+k, r]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if np.max(np.abs(residual)) < tol:
                return U, V, W, True
        
        # Reduce learning rate
        lr *= 0.9
    
    # Final check
    T_approx = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_approx[i,j,k] = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
    
    converged = np.max(np.abs(T_approx - 1.0)) < tol
    return U, V, W, converged

def _try_rank(n, m, p, rank, time_limit, seed_base=0):
    """Try to find a valid decomposition of given rank."""
    start_time = time.time()
    
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        
        seed = seed_base + attempt * 1000
        U, V, W = _random_init(n, m, p, rank, seed=seed)
        
        # Try refinement
        U_r, V_r, W_r, converged = _refine(U, V, W, n, m, p, max_iter=100, tol=1e-8)
        
        if converged:
            # Verify
            valid, err = _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-6)
            if valid:
                return U_r, V_r, W_r
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Leave 10s safety margin
    
    # Start with schoolbook as fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # The theoretical minimum rank for (n,m,p) matmul is at least max(n*m, m*p, n*p)
    # For 3,4,8: max(12, 32, 24) = 32
    # Known bounds: rank >= max(n*m, m*p, n*p), and for square n x n x n, rank >= n^2
    
    min_possible = max(n*m, m*p, n*p)
    
    # Try ranks from min_possible upward
    for rank in range(min_possible, min(schoolbook_rank, 60)):
        if time.time() > deadline:
            break
        
        time_left = deadline - time.time()
        if time_left < 5:
            break
        
        result = _try_rank(n, m, p, rank, time_left * 0.8, seed_base=rank)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = rank
            best_result = {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
            # If we found a good rank, try to improve further
            continue
    
    return best_result
# EVOLVE-BLOCK-END
