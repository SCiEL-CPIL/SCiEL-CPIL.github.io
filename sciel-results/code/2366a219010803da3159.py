"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

import numpy as np
import time

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using NumPy tensor operations."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Compute T = sum_r U[:,r] * V[:,r] * W[:,r]
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check against expected tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    return np.allclose(T, T_target, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    
    best_rank = n * m * p  # schoolbook
    best_decomp = None
    
    # Generate schoolbook decomposition as fallback
    U_sb = [[0] * best_rank for _ in range(n * m)]
    V_sb = [[0] * best_rank for _ in range(m * p)]
    W_sb = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Build target tensor once
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Strategy 1: Try embedding into 6x6 multiplication
    # The 2x3 * 3x6 can be seen as part of a 6x6 * 6x6 multiplication
    # Use known rank-20 decomposition for 6x6 (Strassen-based)
    # For now, try structured initialization based on block structure
    
    # Strategy 2: Random search with better initialization and more attempts
    for target_rank in range(20, 25):  # try ranks from 20 to 24
        if time.time() > deadline:
            break
        
        found = False
        num_attempts = 5 if target_rank >= 22 else 3
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            np.random.seed(1000 + target_rank * 1000 + attempt)
            
            # Try different initialization strategies
            if attempt % 2 == 0:
                # Random initialization
                U = np.random.randn(n*m, target_rank) * 0.3
                V = np.random.randn(m*p, target_rank) * 0.3
                W = np.random.randn(p*n, target_rank) * 0.3
            else:
                # Sparse initialization with some structure
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                for r in range(target_rank):
                    # Randomly assign non-zero entries
                    for a in range(n*m):
                        if np.random.rand() < 0.3:
                            U[a, r] = np.random.choice([-1, 1])
                    for b in range(m*p):
                        if np.random.rand() < 0.3:
                            V[b, r] = np.random.choice([-1, 1])
                    for c in range(p*n):
                        if np.random.rand() < 0.3:
                            W[c, r] = np.random.choice([-1, 1])
            
            # Adaptive learning rate optimization
            lr = 0.05
            best_err = np.inf
            stuck_count = 0
            
            for step in range(300):
                if time.time() > deadline:
                    break
                
                # Compute current tensor
                T_curr = np.einsum('ar,br,cr->abc', U, V, W)
                residual = T_target - T_curr
                err = np.linalg.norm(residual)
                
                if err < best_err:
                    best_err = err
                    stuck_count = 0
                else:
                    stuck_count += 1
                
                if err < 1e-10:
                    break
                
                # Adaptive learning rate
                if stuck_count > 50:
                    lr *= 0.5
                    stuck_count = 0
                    if lr < 1e-5:
                        break
                
                # Gradients
                dU = np.einsum('abc,br,cr->ar', residual, V, W)
                dV = np.einsum('abc,ar,cr->br', residual, U, W)
                dW = np.einsum('abc,ar,br->cr', residual, U, V)
                
                # Update with momentum-like behavior
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Check convergence and try rounding
                if err < 1e-8:
                    # Try to round to exact values
                    for mult in [1, 2, 4, 8]:
                        U_r = np.round(U * mult) / mult
                        V_r = np.round(V * mult) / mult
                        W_r = np.round(W * mult) / mult
                        
                        if _verify_fast(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                            found = True
                            best_rank = target_rank
                            best_decomp = (U_r.tolist(), V_r.tolist(), W_r.tolist())
                            break
                    if found:
                        break
            
            if found:
                break
        
        if found:
            break
    
    U, V, W = best_decomp
    return {"U": U, "V": V, "W": W}
