"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # 115 seconds, leave 5 seconds margin
    
    import random
    import numpy as np
    from fractions import Fraction
    
    def verify(U, V, W):
        # Verify the decomposition
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    def round_to_halfint(x):
        return round(2*x) / 2.0
    
    def is_valid_halfint(U, V, W):
        for row in U:
            for x in row:
                if abs(2*x - round(2*x)) > 1e-9:
                    return False
        for row in V:
            for x in row:
                if abs(2*x - round(2*x)) > 1e-9:
                    return False
        for row in W:
            for x in row:
                if abs(2*x - round(2*x)) > 1e-9:
                    return False
        return True
    
    def compute_error(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    err += (s - 1.0) ** 2
        return err
    
    def coordinate_descent(U, V, W, max_iters=5000, tol=1e-8):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            err = compute_error(U, V, W)
            if err < tol:
                # Round to half-integers
                U = [[round_to_halfint(x) for x in row] for row in U]
                V = [[round_to_halfint(x) for x in row] for row in V]
                W = [[round_to_halfint(x) for x in row] for row in W]
                if verify(U, V, W):
                    return U, V, W, 0.0
                else:
                    # Rounding broke it, try without rounding
                    break
            
            # Update U
            for idx in range(n * m):
                for r in range(rank):
                    # Compute gradient with respect to U[idx][r]
                    grad = 0.0
                    for i in range(n):
                        for j in range(m):
                            if i * m + j == idx:
                                for k in range(p):
                                    s = 0.0
                                    for rr in range(rank):
                                        s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                                    grad += 2 * (s - 1.0) * V[j*p+k][r] * W[k*n+i][r]
                    # Simple gradient step
                    U[idx][r] -= 0.01 * grad
            
            # Update V
            for idx in range(m * p):
                for r in range(rank):
                    grad = 0.0
                    for j in range(m):
                        for k in range(p):
                            if j * p + k == idx:
                                for i in range(n):
                                    s = 0.0
                                    for rr in range(rank):
                                        s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                                    grad += 2 * (s - 1.0) * U[i*m+j][r] * W[k*n+i][r]
                    V[idx][r] -= 0.01 * grad
            
            # Update W
            for idx in range(p * n):
                for r in range(rank):
                    grad = 0.0
                    for k in range(p):
                        for i in range(n):
                            if k * n + i == idx:
                                for j in range(m):
                                    s = 0.0
                                    for rr in range(rank):
                                        s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                                    grad += 2 * (s - 1.0) * U[i*m+j][r] * V[j*p+k][r]
                    W[idx][r] -= 0.01 * grad
        
        err = compute_error(U, V, W)
        return U, V, W, err
    
    def structured_init(rank, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with small random values
        U = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.uniform(-1, 1) for _ in range(rank)] for _ in range(p * n)]
        
        return U, V, W
    
    # Try ranks from lower bound up to 24
    # Lower bound: max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    best_result = None
    best_rank = n * m * p  # 24
    
    for rank in range(12, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_seeds = 20 if rank < 18 else 50
        for seed in range(max_seeds):
            if time.time() > deadline - 10:
                break
            
            U, V, W = structured_init(rank, seed)
            U, V, W, err = coordinate_descent(U, V, W, max_iters=3000 if rank < 18 else 5000)
            
            if err < 1e-8:
                # Found a valid decomposition!
                best_result = (U, V, W)
                best_rank = rank
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
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
# EVOLVE-BLOCK-END
