import numpy as np
import time
import itertools
import sys

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # The matrix multiplication tensor T[i,j,k] for (i in 0..n-1, j in 0..m-1, k in 0..p-1)
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Convention: W[k*n + i, r]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Schoolbook decomposition as baseline
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U0, V0, W0, rank0 = schoolbook()
    best_U, best_V, best_W, best_rank = U0, V0, W0, rank0
    
    def verify(U, V, W, tol=1e-9):
        """Check if decomposition is exact (within tolerance)."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val) > tol:
                        return False
        return True
    
    def try_reduce_rank(U, V, W, target_rank):
        """Try to reduce the rank by finding linear dependencies."""
        if target_rank >= U.shape[1]:
            return None, None, None, None
        
        # Check if we can find a linear combination that works
        # This is a complex optimization problem
        return None, None, None, None
    
    # Strategy: Use the structure of the problem
    # For 3x3 x 3x8, we can think of it as 8 independent 3x3 multiplications
    # but with shared structure in the first factor
    
    # Key insight: The tensor for 3x3 x 3x8 can be decomposed more efficiently
    # by exploiting the fact that the 8 columns of B are independent
    
    # Let's try a direct optimization approach using the tensor structure
    
    # The multiplication tensor has entries T[i,j,k] = 1 for all (i,j,k)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is equivalent to finding a low-rank decomposition of the tensor
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let's use a randomized approach with verification
    
    def random_decomposition(rank, seed=None):
        """Generate a random decomposition and try to optimize it."""
        if seed is not None:
            np.random.seed(seed)
        
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        return U, V, W
    
    def optimize_decomposition(U, V, W, max_iter=1000, lr=0.01):
        """Optimize the decomposition using gradient descent."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iter):
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] should be 1
                        # d/dU[i*m+j, r] = V[j*p+k, r] * W[k*n+i, r]
                        for r in range(U.shape[1]):
                            grad_U[i*m+j, r] += V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += U[i*m+j, r] * W[k*n+i, r]
                            grad_W[k*n+i, r] += U[i*m+j, r] * V[j*p+k, r]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if it % 100 == 0:
                loss = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            loss += (val - 1) ** 2
                if loss < 1e-12:
                    break
        
        return U, V, W
    
    # Try different ranks starting from lower values
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But for exact decomposition, we likely need more
    
    # Let's try ranks from 24 upward
    for target_rank in range(24, rank0 + 1):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            
            U, V, W = random_decomposition(target_rank, seed=42 + attempt)
            
            # Quick check: can we even achieve rank target_rank?
            # The tensor needs to have rank <= target_rank
            
            # Try optimization
            U_opt, V_opt, W_opt = optimize_decomposition(U, V, W, max_iter=500, lr=0.005)
            
            if verify(U_opt, V_opt, W_opt):
                best_U, best_V, best_W, best_rank = U_opt, V_opt, W_opt, target_rank
                break
        
        if best_rank == target_rank:
            break
    
    # If we found a better decomposition, use it; otherwise use schoolbook
    if best_rank < rank0:
        U_final, V_final, W_final = best_U, best_V, best_W
    else:
        U_final, V_final, W_final = U0, V0, W0
    
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
