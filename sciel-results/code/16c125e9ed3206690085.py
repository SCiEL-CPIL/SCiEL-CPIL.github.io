"""Matrix multiplication 2,2,9 rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def _build_target_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    T_recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(rank):
        T_recon += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    T_target = _build_target_tensor(n, m, p)
    diff = np.abs(T_recon - T_target)
    max_err = np.max(diff)
    return max_err < tol, max_err

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Try to find a rank-32 decomposition using Strassen-inspired structure
    # For 2x2*2x2, Strassen uses 7 products.
    # For 2x2*2x9, we can potentially use a similar structure.
    
    # Let's try a systematic search with gradient descent for various ranks
    target = _build_target_tensor(n, m, p)
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Try ranks from 32 down to 28
    for rank in range(32, 27, -1):
        if time.time() > deadline - 10:
            break
        
        np.random.seed(rank * 7919 + 13)
        
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize
        for r in range(rank):
            norm = np.sqrt(np.sum(U[:,r]**2) * np.sum(V[:,r]**2) * np.sum(W[:,r]**2))
            if norm > 1e-12:
                U[:,r] /= norm**(1/3)
                V[:,r] /= norm**(1/3)
                W[:,r] /= norm**(1/3)
        
        lr = 0.05
        best_err = float('inf')
        patience = 0
        
        for step in range(2000):
            if time.time() > deadline - 5:
                break
            
            T_recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
            for r in range(rank):
                T_recon += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
            
            diff = T_recon - target
            err = np.sum(diff**2)
            
            if err < best_err:
                best_err = err
                patience = 0
            else:
                patience += 1
            
            if best_err < 1e-14:
                break
            if patience > 200:
                break
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r in range(rank):
                grad_U[:,r] = np.einsum('ibc,br,cr->i', diff, V[:,r], W[:,r])
                grad_V[:,r] = np.einsum('ibc,ir,cr->b', diff, U[:,r], W[:,r])
                grad_W[:,r] = np.einsum('ibc,ir,br->c', diff, U[:,r], V[:,r])
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            if err > best_err * 1.5:
                lr *= 0.5
            elif err < best_err * 0.8:
                lr = min(lr * 1.1, 0.2)
        
        if best_err < 1e-12:
            # Verify with the actual matrix representation
            U_list = [[float(U[i,r]) for r in range(rank)] for i in range(n*m)]
            V_list = [[float(V[i,r]) for r in range(rank)] for i in range(m*p)]
            W_list = [[float(W[i,r]) for r in range(rank)] for i in range(p*n)]
            
            valid, max_err = _verify_decomposition(U, V, W, n, m, p)
            if valid:
                best_result = (U_list, V_list, W_list, rank)
                break
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition of rank 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
