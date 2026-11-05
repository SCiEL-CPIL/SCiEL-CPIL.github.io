"""Matrix multiplication 3x3 by 3x7 - rank minimization."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Try known fast matrix multiplication algorithms
    # Strassen's algorithm gives rank 7 for 2x2x2
    # We can use blocking to extend to larger matrices
    
    # For 3x3 * 3x7, we can potentially use:
    # - Block decomposition using Strassen on 2x2 blocks
    # - Or try rank minimization via optimization
    
    def verify(U, V, W, rank):
        """Check if the decomposition is valid."""
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        expected = 1.0 if j1 == j2 else 0.0
                        actual = 0.0
                        for c in range(rank):
                            actual += U[i * m + j1, c] * V[j2 * p + k, c] * W[k * n + i, c]
                        if abs(actual - expected) > 1e-6:
                            return False
        return True

    def loss(params, target_rank):
        """Loss function for optimization: sum of squared errors."""
        U = params[:n * m * target_rank].reshape(n * m, target_rank)
        V = params[n * m * target_rank : n * m * target_rank + m * p * target_rank].reshape(m * p, target_rank)
        W = params[n * m * target_rank + m * p * target_rank :].reshape(p * n, target_rank)
        
        total_loss = 0.0
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        expected = 1.0 if j1 == j2 else 0.0
                        actual = 0.0
                        for c in range(target_rank):
                            actual += U[i * m + j1, c] * V[j2 * p + k, c] * W[k * n + i, c]
                        total_loss += (actual - expected) ** 2
        return total_loss

    # Try to find lower rank decompositions using optimization
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    # Try ranks from schoolbook-1 down to a reasonable minimum
    # For 3x3*3x7, the theoretical lower bound is at least max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    min_rank_to_try = 25  # Start trying from a reasonable lower bound
    
    for target_rank in range(rank_school - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple random initializations
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
            
            # Initialize with small random perturbations
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Use coordinate descent / simple gradient steps
            # We'll do a simple iterative refinement
            lr = 0.01
            for step in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for k in range(p):
                        for j1 in range(m):
                            for j2 in range(m):
                                expected = 1.0 if j1 == j2 else 0.0
                                actual = 0.0
                                for c in range(target_rank):
                                    actual += U[i * m + j1, c] * V[j2 * p + k, c] * W[k * n + i, c]
                                err = actual - expected
                                for c in range(target_rank):
                                    dU[i * m + j1, c] += 2 * err * V[j2 * p + k, c] * W[k * n + i, c]
                                    dV[j2 * p + k, c] += 2 * err * U[i * m + j1, c] * W[k * n + i, c]
                                    dW[k * n + i, c] += 2 * err * U[i * m + j1, c] * V[j2 * p + k, c]
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Compute current loss
                current_loss = 0.0
                for i in range(n):
                    for k in range(p):
                        for j1 in range(m):
                            for j2 in range(m):
                                expected = 1.0 if j1 == j2 else 0.0
                                actual = 0.0
                                for c in range(target_rank):
                                    actual += U[i * m + j1, c] * V[j2 * p + k, c] * W[k * n + i, c]
                                current_loss += (actual - expected) ** 2
                
                if current_loss < 1e-8:
                    if verify(U, V, W, target_rank):
                        best_rank = target_rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        break
            if best_rank == target_rank:
                break
    
    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
