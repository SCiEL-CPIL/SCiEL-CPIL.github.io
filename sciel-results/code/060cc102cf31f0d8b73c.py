"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0  # The matrix multiplication tensor has 1s where indices match
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    def random_search(target_rank, max_iters=1000):
        """Try to find a decomposition of given rank using random initialization and refinement."""
        best_result = None
        
        for attempt in range(max_iters):
            if time.time() > deadline:
                break
                
            # Random initialization with small values
            U = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p*n)]
            
            # Try a few gradient descent steps to see if we can converge
            lr = 0.01
            for step in range(50):
                if time.time() > deadline:
                    break
                
                # Compute gradients
                dU = [[0.0] * target_rank for _ in range(n*m)]
                dV = [[0.0] * target_rank for _ in range(m*p)]
                dW = [[0.0] * target_rank for _ in range(p*n)]
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                            # dT/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c]
                            # dT/dV[j*p+k,c] = U[i*m+j,c] * W[k*n+i,c]
                            # dT/dW[k*n+i,c] = U[i*m+j,c] * V[j*p+k,c]
                            for c in range(target_rank):
                                u_val = U[i*m+j][c]
                                v_val = V[j*p+k][c]
                                w_val = W[k*n+i][c]
                                grad = (u_val * v_val * w_val - 1.0)
                                dU[i*m+j][c] += grad * v_val * w_val
                                dV[j*p+k][c] += grad * u_val * w_val
                                dW[k*n+i][c] += grad * u_val * v_val
                
                # Update
                for idx in range(n*m):
                    for c in range(target_rank):
                        U[idx][c] -= lr * dU[idx][c]
                for idx in range(m*p):
                    for c in range(target_rank):
                        V[idx][c] -= lr * dV[idx][c]
                for idx in range(p*n):
                    for c in range(target_rank):
                        W[idx][c] -= lr * dW[idx][c]
            
            # Check if we found a valid decomposition
            if verify(U, V, W, target_rank):
                return U, V, W, target_rank
        
        return None
    
    # Start with schoolbook as fallback
    U, V, W, rank = schoolbook()
    
    # Try to find lower rank decompositions
    # Known lower bounds: for 3x5x6, the rank is at least max(3*5, 5*6, 3*6) = 30
    # Try ranks from 30 up to 89
    for target_rank in range(30, 90):
        if time.time() > deadline:
            break
        
        result = random_search(target_rank, max_iters=20)
        if result is not None:
            U, V, W, rank = result
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
