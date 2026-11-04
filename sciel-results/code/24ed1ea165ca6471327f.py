"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try optimization for lower rank
    for rank in [80, 90, 100, 110]:
        if time.time() > deadline - 20:
            break
        
        for seed in range(10):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(seed * 31 + rank)
            U = np.random.randn(n*m, rank) * 0.3
            V = np.random.randn(m*p, rank) * 0.3
            W = np.random.randn(p*n, rank) * 0.3
            
            # Simple gradient descent
            lr = 0.02
            for step in range(2000):
                if time.time() > deadline - 15:
                    break
                
                T_est = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                err = T - T_est
                loss = np.sum(err**2)
                
                if loss < 1e-10:
                    break
                
                grad_U = np.einsum('abc,br,cr->ar', err, V, W, optimize=True)
                grad_V = np.einsum('abc,ar,cr->br', err, U, W, optimize=True)
                grad_W = np.einsum('abc,ar,br->cr', err, U, V, optimize=True)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            T_est = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            loss = np.sum((T - T_est)**2)
            
            if loss < 1e-8:
                U_s = np.round(U * 2) / 2
                V_s = np.round(V * 2) / 2
                W_s = np.round(W * 2) / 2
                
                T_s = np.einsum('ar,br,cr->abc', U_s, V_s, W_s, optimize=True)
                loss_s = np.sum((T - T_s)**2)
                
                if loss_s < 1e-8:
                    best_rank = rank
                    best_U, best_V, best_W = U_s, V_s, W_s
                    break
        
        if best_rank < rank_sb:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
