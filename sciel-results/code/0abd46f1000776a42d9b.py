"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # The multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Index conventions:
    #   U row index: i*m + j  (i in 0..n-1, j in 0..m-1)
    #   V row index: j*p + k  (j in 0..m-1, k in 0..p-1)
    #   W row index: k*n + i  (k in 0..p-1, i in 0..n-1)
    
    # Schoolbook rank = n*m*p = 54
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Function to compute error tensor efficiently using vectorization
    def compute_error(U, V, W, rank):
        """Compute error[i,j,k] = 1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
        # Reshape for efficient computation
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        error = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    error[i,j,k] = 1.0 - np.dot(u_val, v_val * w_val)
        return error
    
    # Function to check if all errors are below tolerance
    def is_valid(error, tol=1e-9):
        return np.all(np.abs(error) < tol)
    
    # Schoolbook decomposition
    def schoolbook_decomp():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Strategy: Use Strassen-like approach for 3x3x3 blocks
    # First, construct a rank-23 Strassen decomposition for 3x3x3
    # Then extend to 3x3x6 by treating the 6 columns as two 3-column blocks
    
    # Strassen decomposition for 3x3 matrix multiplication (rank 23)
    # This is a known decomposition; we'll construct it explicitly
    def strassen_3x3():
        """Return rank-23 decomposition for 3x3x3 matrix multiplication."""
        # Using the known Strassen algorithm for 3x3 matrices
        # The standard Strassen algorithm gives rank 7 for 2x2x2
        # For 3x3x3, we can use a combination of Strassen and direct methods
        
        # Actually, let's use a simpler approach: 
        # For 3x3x3, the schoolbook rank is 27
        # We can try to find a lower rank decomposition
        
        # Let's use a known result: 3x3x3 matrix multiplication has rank at most 23
        # We'll construct it using the "laser method" or explicit construction
        
        # For now, let's use a structured approach:
        # Split 3x3 into blocks and use Strassen where possible
        
        # Actually, let's just use schoolbook for 3x3x3 (rank 27) and then
        # optimize the 3x3x6 case directly
        
        rank = 27
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # For 3x3x3 schoolbook
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        return U, V, W, rank
    
    # Try to find a low-rank decomposition for 3x3x6 directly
    # Using gradient-based optimization with multiple restarts
    
    # Candidate ranks to try (from low to high)
    # Based on known results, 3x3x6 should have rank around 30-40
    candidate_ranks = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 40, 42, 44, 46, 48, 50, 52]
    
    # For each target rank, try multiple random initializations
    max_init_per_rank = 3
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        for init_idx in range(max_init_per_rank):
            if time.time() > deadline:
                break
                
            # Initialize with random values
            np.random.seed(42 + target_rank * 100 + init_idx)
            U = np.random.randn(n*m, target_rank) * 0.3
            V = np.random.randn(m*p, target_rank) * 0.3
            W = np.random.randn(p*n, target_rank) * 0.3
            
            # Iterative refinement using gradient descent with adaptive step size
            eta = 0.1
            min_eta = 1e-6
            for iteration in range(300):
                if time.time() > deadline:
                    break
                    
                # Compute error
                error = compute_error(U, V, W, target_rank)
                max_err = np.max(np.abs(error))
                
                # Check if valid
                if max_err < 1e-9:
                    best_rank = target_rank
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    break
                
                # Compute gradients using vectorized operations
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            err = error[i,j,k]
                            if abs(err) < 1e-12:
                                continue
                            u_idx = i*m+j
                            v_idx = j*p+k
                            w_idx = k*n+i
                            dU[u_idx, :] += err * (V[v_idx, :] * W[w_idx, :])
                            dV[v_idx, :] += err * (U[u_idx, :] * W[w_idx, :])
                            dW[w_idx, :] += err * (U[u_idx, :] * V[v_idx, :])
                
                # Adaptive learning rate based on error magnitude
                if max_err > 0.1:
                    eta = max(eta * 0.9, min_eta)
                elif max_err < 0.01:
                    eta = min(eta * 1.1, 0.5)
                
                # Update
                U -= eta * dU
                V -= eta * dV
                W -= eta * dW
            
            if best_rank <= target_rank:
                break
    
    # If no better decomposition found, use schoolbook
    if best_U is None:
        best_U, best_V, best_W, best_rank = schoolbook_decomp()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
