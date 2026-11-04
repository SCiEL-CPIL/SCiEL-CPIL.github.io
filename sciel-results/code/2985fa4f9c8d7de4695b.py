"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U,V,W decompose the matrix multiplication tensor."""
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    rank = U.shape[1]
    # Build the tensor and check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if j == k else 0.0
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - expected) > tol:
                    return False
    return True

def _try_random(rank, n, m, p, seed=0, max_iter=50000):
    """Try to find a rank-r decomposition via random search."""
    rng = np.random.RandomState(seed)
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    for it in range(max_iter):
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0 if j == k else 0.0
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (val - expected) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if err < 1e-4:
            break
    
    return best_err, best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 75
    
    deadline = time.time() + 140  # safety margin
    
    # Try various ranks from low to high
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = 25 for this case
    # But actually the tensor rank lower bound is at least max(n, m, p) * something
    # Let's try ranks from 25 upward
    
    best_result = None
    best_rank = schoolbook_rank
    
    # First, try to find a decomposition with rank < schoolbook
    for rank in range(25, schoolbook_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random seeds
        found = False
        for seed in range(20):
            if time.time() > deadline:
                break
            
            err, U, V, W = _try_random(rank, n, m, p, seed=seed, max_iter=10000)
            
            if err < 1e-4:
                # Verify exactly
                U_list = [[float(U[i,j]) for j in range(rank)] for i in range(n*m)]
                V_list = [[float(V[i,j]) for j in range(rank)] for i in range(m*p)]
                W_list = [[float(W[i,j]) for j in range(rank)] for i in range(p*n)]
                
                if _verify(U_list, V_list, W_list, n, m, p, tol=1e-5):
                    found = True
                    best_rank = rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break
        
        if found:
            # Try to go even lower
            if rank > 25:
                continue
            else:
                break
    
    if best_result is not None:
        return best_result
    
    # Fall back to schoolbook
    rank = schoolbook_rank
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
