"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Schoolbook decomposition: rank = n*m*p = 80
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = len(U[0])
        # Build the multiplication tensor T[i,j,k] = U[i,j] @ V[j,k] @ W[k,i]
        # Actually: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if i,j,k valid, else 0
        
        # The tensor for matrix multiplication (n x m) * (m x p) -> (n x p)
        # T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k (it's the standard matmul tensor)
        
        # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(r):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_rank(r, max_iterations=2000, seed=42):
        """Try to find a rank-r decomposition using randomized local search."""
        if time.time() - start_time > DEADLINE - 5:
            return None
        
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iterations):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Compute loss: sum over all (i,j,k) of (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(r):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        loss += (s - 1.0) ** 2
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_loss < 1e-20:
                # Found exact solution
                break
            
            # Gradient descent step
            lr = 0.01 / (1 + it * 0.001)
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # s = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        err = 2.0 * (s - 1.0)
                        
                        for r in range(r):
                            dU[i*m+j, r] += err * V[j*p+k, r] * W[k*n+i, r]
                            dV[j*p+k, r] += err * U[i*m+j, r] * W[k*n+i, r]
                            dW[k*n+i, r] += err * U[i*m+j, r] * V[j*p+k, r]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_loss < 1e-20:
            # Convert to lists with rounded values
            U_list = [[round(float(x), 6) for x in row] for row in best_U]
            V_list = [[round(float(x), 6) for x in row] for row in best_V]
            W_list = [[round(float(x), 6) for x in row] for row in best_W]
            
            if verify_decomposition(U_list, V_list, W_list, tol=1e-8):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 20 upward (known lower bound is around 16-20 for 4x4x5)
    for r in range(20, min(rank_school, 40)):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        result = try_rank(r, max_iterations=3000, seed=r)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
