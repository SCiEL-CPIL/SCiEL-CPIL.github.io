import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s margin within 120s limit
    
    def time_check():
        return time.time() < deadline
    
    # The tensor for matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # Rank-r decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # (using trace convention W[k*n+i, r])
    
    # Strategy: Use block decomposition with Strassen-style tricks.
    # For 4x5 * 5x6, split into blocks and use known efficient decompositions.
    
    # First, let's try a systematic approach:
    # 1. Start with trivial rank-120
    # 2. Try to reduce by finding linear dependencies
    
    # Actually, let's use a more principled approach:
    # The matrix multiplication tensor M_{n,m,p} has known rank bounds.
    # For small matrices, we can use exhaustive or greedy search.
    
    # Key insight: We can formulate this as a constraint satisfaction problem.
    # Given rank r, we need to find U, V, W such that:
    # sum_{r_idx} U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx] = 1 for all i,j,k
    
    # This is a system of polynomial equations. Let's use a greedy approach:
    # Start with random initializations and use gradient descent on a loss function.
    
    def loss_function(U, V, W, n, m, p):
        """Compute the loss for a given decomposition."""
        total_loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    total_loss += (val - 1.0) ** 2
        return total_loss
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try ranks from a reasonable lower bound up to 120
    # Lower bound: max(n*p, n*m, m*p) = max(24, 20, 30) = 30
    # But realistically, for 4x5*5x6, the rank is likely between 50-80
    
    best_result = None
    best_rank = n * m * p  # trivial
    
    # Initialize with trivial decomposition
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
    
    best_result = (U_trivial.copy(), V_trivial.copy(), W_trivial.copy())
    
    # Try to find lower rank decompositions using optimization
    # We'll try ranks from 60 down to 40 (reasonable range)
    
    np.random.seed(42)
    
    for target_rank in range(70, 35, -1):
        if not time_check():
            break
        
        # Try multiple random initializations for this rank
        for attempt in range(5):
            if not time_check():
                break
            
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Use gradient descent with learning rate schedule
            lr = 0.01
            for step in range(500):
                if not time_check():
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err = val - 1.0
                            for r in range(target_rank):
                                dU[i*m+j, r] += 2 * err * V[j*p+k, r] * W[k*n+i, r]
                                dV[j*p+k, r] += 2 * err * U[i*m+j, r] * W[k*n+i, r]
                                dW[k*n+i, r] += 2 * err * U[i*m+j, r] * V[j*p+k, r]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Decay learning rate
                if step % 100 == 0:
                    lr *= 0.5
                
                # Check convergence
                current_loss = loss_function(U, V, W, n, m, p)
                if current_loss < 1e-10:
                    break
            
            # Verify if we found a valid decomposition
            if verify_decomposition(U, V, W, n, m, p):
                best_result = (U.copy(), V.copy(), W.copy())
                best_rank = target_rank
                break  # Found a valid decomposition for this rank, try lower
        
        if not time_check():
            break
    
    # If we found a better decomposition, use it
    if best_rank < rank_trivial:
        U, V, W = best_result
    else:
        U, V, W = best_result
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
