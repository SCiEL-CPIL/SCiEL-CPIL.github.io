"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 8]

def verify_decomposition(U, V, W, n=4, m=4, p=8, trials=100):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        
        # Compute using decomposition
        C_comp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for r in range(rank):
                        C_comp[i, k] += A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        if not np.allclose(C_true, C_comp, atol=1e-9):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try to reduce rank using random perturbation and projection
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Try different target ranks
        target_rank = best_rank - random.randint(1, min(10, best_rank - 1))
        if target_rank < 1:
            continue
        
        # Random initialization
        U_new = np.random.randn(n*m, target_rank)
        V_new = np.random.randn(m*p, target_rank)
        W_new = np.random.randn(p*n, target_rank)
        
        # Use gradient descent to minimize the error
        lr = 0.01
        for step in range(100):
            if time.time() > deadline:
                break
            
            # Compute error
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(target_rank):
                            val += U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r]
                        error += (val - 1) ** 2
            
            if error < 1e-12:
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
                best_rank = target_rank
                break
        
        # Check if we found a better solution
        if verify_decomposition(U_new, V_new, W_new, trials=10):
            best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
            best_rank = target_rank
    
    # Convert to lists for return
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
