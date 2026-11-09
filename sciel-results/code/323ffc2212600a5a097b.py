"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Schoolbook decomposition of rank n*m*p
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition using iterative optimization
    # The tensor equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is a system of nonlinear equations. We can use gradient-based optimization
    # to minimize the sum of squared errors.
    
    if time.time() < deadline - 5:
        np.random.seed(42)
        
        # Try various target ranks below schoolbook
        for target_rank in range(44, 20, -1):
            if time.time() > deadline - 5:
                break
            
            # Multiple random restarts
            max_restarts = 20 if target_rank > 35 else 10
            for attempt in range(max_restarts):
                if time.time() > deadline - 5:
                    break
                
                # Random initialization with small values
                U_rand = np.random.randn(n * m, target_rank) * 0.5
                V_rand = np.random.randn(m * p, target_rank) * 0.5
                W_rand = np.random.randn(p * n, target_rank) * 0.5
                
                # Gradient-based optimization
                lr = 0.01
                max_iter = 200
                prev_loss = float('inf')
                
                for it in range(max_iter):
                    if time.time() > deadline - 5:
                        break
                    
                    # Compute loss: sum over all (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
                    loss = 0.0
                    grad_U = np.zeros_like(U_rand)
                    grad_V = np.zeros_like(V_rand)
                    grad_W = np.zeros_like(W_rand)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                # Compute the dot product sum
                                prod_sum = 0.0
                                for c in range(target_rank):
                                    prod_sum += U_rand[i*m+j, c] * V_rand[j*p+k, c] * W_rand[k*n+i, c]
                                
                                err = prod_sum - 1.0
                                loss += err * err
                                
                                # Gradients
                                for c in range(target_rank):
                                    u_val = U_rand[i*m+j, c]
                                    v_val = V_rand[j*p+k, c]
                                    w_val = W_rand[k*n+i, c]
                                    
                                    # d(loss)/d(U[i*m+j, c]) = 2 * err * V[j*p+k, c] * W[k*n+i, c]
                                    grad_U[i*m+j, c] += 2.0 * err * v_val * w_val
                                    grad_V[j*p+k, c] += 2.0 * err * u_val * w_val
                                    grad_W[k*n+i, c] += 2.0 * err * u_val * v_val
                    
                    # Gradient descent step
                    U_rand -= lr * grad_U
                    V_rand -= lr * grad_V
                    W_rand -= lr * grad_W
                    
                    # Check convergence
                    if abs(loss - prev_loss) < 1e-12:
                        break
                    prev_loss = loss
                
                # Verify the result
                if _verify_decomposition(U_rand, V_rand, W_rand, n, m, p):
                    best = {
                        "U": U_rand.tolist(),
                        "V": V_rand.tolist(),
                        "W": W_rand.tolist()
                    }
                    best_rank = target_rank
                    break  # Found a valid decomposition at this rank
    
    return best
# EVOLVE-BLOCK-END
