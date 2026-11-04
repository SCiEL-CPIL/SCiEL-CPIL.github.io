"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    DEADLINE = time.time() + 140.0  # 140s out of 150s budget
    START_TIME = time.time()
    
    # Return a valid schoolbook decomposition if any error occurs
    def schoolbook_decomposition():
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
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check that the tensor decomposes correctly
        # The matrix multiplication tensor T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0  # T[i,j,k] = 1 for all i,j,k
                    rhs = 0.0
                    for c in range(rank):
                        rhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(lhs - rhs) > 1e-10:
                        return False
        return True
    
    def try_decomposition_with_rank(r, num_trials=100, timeout=None):
        """
        Try to find a rank-r decomposition using numerical optimization.
        Uses a simple iterative approach.
        """
        if timeout is None:
            timeout = 60
        
        best_U, best_V, best_W = None, None, None
        
        for trial in range(num_trials):
            if time.time() > timeout:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Scale to encourage integer-like values
            U = np.round(U * 4) / 4.0
            V = np.round(V * 4) / 4.0
            W = np.round(W * 4) / 4.0
            
            # Simple local search
            lr = 0.05
            for it in range(200):
                # Compute residuals for random subset of (i,j,k) triples
                residuals = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            rhs = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            residuals += (rhs - 1.0) ** 2
                
                if residuals < 1e-12:
                    break
                
                # Simple gradient-free update
                for c in range(r):
                    for _ in range(3):
                        idx_U = (random.randint(0, n*m-1), c)
                        idx_V = (random.randint(0, m*p-1), c)
                        idx_W = (random.randint(0, p*n-1), c)
                        
                        # Try small perturbations
                        dU = random.choice([-0.1, 0, 0.1])
                        dV = random.choice([-0.1, 0, 0.1])
                        dW = random.choice([-0.1, 0, 0.1])
                        
                        if dU == 0 and dV == 0 and dW == 0:
                            continue
                        
                        old_res = residuals
                        U[idx_U] += dU
                        V[idx_V] += dV
                        W[idx_W] += dW
                        
                        new_res = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    rhs = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                    new_res += (rhs - 1.0) ** 2
                        
                        if new_res > old_res:
                            U[idx_U] -= dU
                            V[idx_V] -= dV
                            W[idx_W] -= dW
                        else:
                            residuals = new_res
                
                if time.time() > timeout:
                    break
            
            # Check if we found a valid decomposition
            if verify(U.tolist(), V.tolist(), W.tolist()):
                best_U, best_V, best_W = U, V, W
                break
        
        if best_U is not None:
            return {
                "U": best_U.tolist(),
                "V": best_V.tolist(),
                "W": best_W.tolist()
            }
        return None
    
    # Try to find a lower rank decomposition
    # Known lower bounds and known decompositions:
    # For 3x3x3, the best known is rank 23
    # For 3x3x8, we can try various ranks
    
    # Start from a reasonable lower bound and go up
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But the tensor structure may require more
    
    # Try ranks from 24 up to 72 (schoolbook)
    best_result = None
    
    # First, try a few promising ranks with limited time
    trial_ranks = [24, 30, 40, 50, 60, 64, 72]
    
    for r in trial_ranks:
        if time.time() > DEADLINE - 20:
            break
        
        # Limit time per rank attempt
        time_per_rank = min(20, max(5, (DEADLINE - time.time()) / 1.5 - 20))
        
        result = try_decomposition_with_rank(r, num_trials=20, timeout=time_per_rank)
        
        if result is not None:
            best_result = result
            print(f"Found rank-{r} decomposition!")
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    print("Falling back to schoolbook decomposition")
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
