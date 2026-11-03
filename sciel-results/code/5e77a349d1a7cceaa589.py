"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    Strategy:
    1. Precompute the target tensor T
    2. Try ranks from lower bound upward with structured initializations
    3. Use gradient descent with multiple restarts and structured seeds
    4. Validate using vectorized NumPy operations
    5. Return best found or fall back to schoolbook
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    DEADLINE = time.time() + 145  # 145s deadline, 5s safety margin
    
    def schoolbook_decomposition():
        """Return the schoolbook rank n*m*p decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Precompute target tensor
    T = np.zeros((n * m, m * p, p * n))
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                valid_indices.append((a, b, c))
    
    def validate_fast(U, V, W):
        """Fast vectorized validation using einsum."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Compute all entries: E[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        E = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        # Check valid entries are 1
        for (a, b, c) in valid_indices:
            if abs(E[a, b, c] - 1.0) > 1e-9:
                return False
        
        # Check invalid entries are 0
        # Create mask of valid positions
        valid_mask = np.zeros((n * m, m * p, p * n), dtype=bool)
        for (a, b, c) in valid_indices:
            valid_mask[a, b, c] = True
        
        # Check all invalid positions
        invalid_mask = ~valid_mask
        if np.any(np.abs(E[invalid_mask]) > 1e-9):
            return False
        
        return True
    
    def try_rank_r_decomposition(r, max_iter=3000, timeout=20, seed=0):
        """
        Try to find a rank-r decomposition using gradient descent.
        Uses structured initialization based on seed.
        """
        if time.time() >= DEADLINE:
            return None
            
        np.random.seed(seed)
        
        start_time = time.time()
        
        # Different initialization strategies based on seed
        if seed % 3 == 0:
            # Random uniform
            U = np.random.uniform(-1, 1, (n * m, r))
            V = np.random.uniform(-1, 1, (m * p, r))
            W = np.random.uniform(-1, 1, (p * n, r))
        elif seed % 3 == 1:
            # Sparse initialization
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for col in range(r):
                U[np.random.choice(n * m, size=min(3, n * m), replace=False), col] = 1
                V[np.random.choice(m * p, size=min(3, m * p), replace=False), col] = 1
                W[np.random.choice(p * n, size=min(3, p * n), replace=False), col] = 1
        else:
            # Structured based on schoolbook pattern
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for col in range(r):
                i = (col // (m * p)) % n
                j = (col // p) % m
                k = col % p
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                # Add small random perturbation
                U[:, col] += np.random.uniform(-0.1, 0.1, n * m)
                V[:, col] += np.random.uniform(-0.1, 0.1, m * p)
                W[:, col] += np.random.uniform(-0.1, 0.1, p * n)
        
        lr = 0.05
        best_res = float('inf')
        best_U, best_V, best_W = None, None, None
        patience = 0
        max_patience = 200
        
        for it in range(max_iter):
            if time.time() >= DEADLINE or time.time() - start_time > timeout:
                break
            
            # Compute residual and gradients
            E = np.einsum('ar,br,cr->abc', U, V, W)
            R = E - T
            res = np.sum(R**2)
            
            if res < best_res:
                best_res = res
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                patience = 0
            else:
                patience += 1
            
            if patience > max_patience:
                break
            
            if res < 1e-12:
                # Very close to zero, try to round and validate
                U_rounded = [[round(x * 2) / 2 for x in row] for row in best_U.tolist()]
                V_rounded = [[round(x * 2) / 2 for x in row] for row in best_V.tolist()]
                W_rounded = [[round(x * 2) / 2 for x in row] for row in best_W.tolist()]
                
                if validate_fast(U_rounded, V_rounded, W_rounded):
                    return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                
                # Also try integer rounding
                U_int = [[round(x) for x in row] for row in best_U.tolist()]
                V_int = [[round(x) for x in row] for row in best_V.tolist()]
                W_int = [[round(x) for x in row] for row in best_W.tolist()]
                
                if validate_fast(U_int, V_int, W_int):
                    return {"U": U_int, "V": V_int, "W": W_int}
            
            # Compute gradients
            grad_U = 2.0 * np.einsum('abc,br,cr->ar', R, V, W)
            grad_V = 2.0 * np.einsum('abc,ar,cr->br', R, U, W)
            grad_W = 2.0 * np.einsum('abc,ar,br->cr', R, U, V)
            
            # Update with adaptive learning rate
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Decay learning rate
            lr *= 0.998
            
            # Occasionally reset to best to escape local minima
            if it % 500 == 0 and it > 0:
                U = best_U + np.random.uniform(-0.1, 0.1, (n * m, r))
                V = best_V + np.random.uniform(-0.1, 0.1, (m * p, r))
                W = best_W + np.random.uniform(-0.1, 0.1, (p * n, r))
        
        # Final check
        if best_U is not None and best_res < 1e-10:
            U_rounded = [[round(x * 2) / 2 for x in row] for row in best_U.tolist()]
            V_rounded = [[round(x * 2) / 2 for x in row] for row in best_V.tolist()]
            W_rounded = [[round(x * 2) / 2 for x in row] for row in best_W.tolist()]
            
            if validate_fast(U_rounded, V_rounded, W_rounded):
                return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
            
            U_int = [[round(x) for x in row] for row in best_U.tolist()]
            V_int = [[round(x) for x in row] for row in best_V.tolist()]
            W_int = [[round(x) for x in row] for row in best_W.tolist()]
            
            if validate_fast(U_int, V_int, W_int):
                return {"U": U_int, "V": V_int, "W": W_int}
        
        return None
    
    # Try ranks from lower bound upward
    best = None
    
    # Lower bound is max(n*m, m*p, p*n) = 18
    # Try from 18 upward, but focus on reasonable range
    for r in range(18, 54):
        if time.time() >= DEADLINE - 5:  # Leave 5s for return
            break
        
        # Try multiple restarts for this rank
        found = False
        for restart in range(8):
            if time.time() >= DEADLINE - 5:
                break
            
            result = try_rank_r_decomposition(r, max_iter=2000, timeout=15, seed=restart * 1000 + r)
            if result is not None:
                best = result
                found = True
                break
        
        if found:
            break
    
    if best is not None:
        return best
    
    # Fall back to schoolbook
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
