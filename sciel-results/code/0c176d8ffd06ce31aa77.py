"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 115  # Leave some buffer
    
    # The tensor T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def loss_and_grad(U, V, W):
        """Compute loss and gradients."""
        loss = 0.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    diff = val - 1.0
                    loss += diff * diff
                    
                    # Gradient w.r.t U[i*m+j, c]
                    dU[i*m+j, :] += 2.0 * diff * V[j*p+k, :] * W[k*n+i, :]
                    # Gradient w.r.t V[j*p+k, c]
                    dV[j*p+k, :] += 2.0 * diff * U[i*m+j, :] * W[k*n+i, :]
                    # Gradient w.r.t W[k*n+i, c]
                    dW[k*n+i, :] += 2.0 * diff * U[i*m+j, :] * V[j*p+k, :]
                    
        return loss, dU, dV, dW
    
    def optimize(U, V, W, max_iters=500, lr=0.1, time_limit=100):
        """Run gradient descent with adaptive learning rate."""
        prev_loss = float('inf')
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            loss, dU, dV, dW = loss_and_grad(U, V, W)
            if loss < 1e-10:
                return U, V, W, loss
            
            # Adaptive learning rate
            if loss > prev_loss * 1.05:
                lr *= 0.5
            elif it % 10 == 0 and loss < prev_loss * 0.9:
                lr *= 1.1
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            prev_loss = loss
        
        return U, V, W, loss
    
    # Try ranks from low to high, starting from theoretical lower bound
    # For 3x3x4, lower bound is at least 23 (from 3x3x3 results)
    # Try: 23, 24, 25, ..., up to 35 (schoolbook is 36)
    
    best_solution = None
    best_rank = n * m * p  # schoolbook
    
    # First, create schoolbook as fallback
    U_school = np.zeros((n * m, best_rank))
    V_school = np.zeros((m * p, best_rank))
    W_school = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    for rank in range(23, 36):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Multiple random restarts
        for attempt in range(3):
            if time.time() - start_time > time_limit - 5:
                break
            
            np.random.seed(rank * 1000 + attempt)
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            U, V, W, final_loss = optimize(U, V, W, max_iters=300, lr=0.1, time_limit=time_limit - 5)
            
            if final_loss < 1e-8:
                if verify(U, V, W, tol=1e-6):
                    best_solution = (U, V, W)
                    best_rank = rank
                    break
        
        if best_solution is not None:
            break
    
    if best_solution is not None:
        U, V, W = best_solution
    else:
        U, V, W = U_school, V_school, W_school
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
