"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

import numpy as np
import time

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for c in range(rank):
                    lhs += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                # T[i,j,k] should be 1 if this is the standard tensor
                # Actually, the tensor T[i,j,k] = 1 for all i,j,k in the multiplication tensor
                if abs(lhs - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_decomposition(n, m, p, rank, max_time=10.0, seed=42):
    """Try to find a rank decomposition using randomized algebraic method."""
    start = time.time()
    rng = np.random.RandomState(seed)
    
    # Initialize randomly
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    # Use Gauss-Newton-like iteration
    for iteration in range(2000):
        if time.time() - start > max_time:
            break
        
        # Compute residuals
        residuals = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    residuals[i, j, k] = val - 1.0
        
        err = np.linalg.norm(residuals)
        if err < 1e-10:
            # Convert to list of lists
            U_list = [[U[i, c] for c in range(rank)] for i in range(n*m)]
            V_list = [[V[i, c] for c in range(rank)] for i in range(m*p)]
            W_list = [[W[i, c] for c in range(rank)] for i in range(p*n)]
            return U_list, V_list, W_list
        
        # Simple gradient step (approximate)
        # This is a very rough approach
        lr = 0.01 / (1 + iteration)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    dU = residuals[i, j, k] * V[j*p+k] * W[k*n+i]
                    dV = residuals[i, j, k] * U[i*m+j] * W[k*n+i]
                    dW = residuals[i, j, k] * U[i*m+j] * V[j*p+k]
                    U[i*m+j] -= lr * dU
                    V[j*p+k] -= lr * dV
                    W[k*n+i] -= lr * dW
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Safety margin
    
    # Try various ranks
    for rank in [40, 42, 44, 46, 48, 50, 52, 54]:
        if time.time() > deadline - 20:
            break
        result = _try_rank_decomposition(n, m, p, rank, max_time=15.0, seed=rank*1000)
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
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
