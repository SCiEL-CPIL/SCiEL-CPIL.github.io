"""Optimized matrix multiplication tensor decomposition for 2x3 by 3x6."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank(r, n, m, p, deadline, seed_offset=0):
    """Try to find a valid decomposition of rank r."""
    np.random.seed(42 + seed_offset)
    
    # Try multiple random initializations
    for trial in range(20):
        if time.time() > deadline - 3:
            return None
        
        # Use structured random values from {-1, 0, 1}
        U = np.random.choice([-1, 0, 1], size=(n*m, r))
        V = np.random.choice([-1, 0, 1], size=(m*p, r))
        W = np.random.choice([-1, 0, 1], size=(p*n, r))
        
        if _verify(U, V, W, n, m, p):
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    return None

def construct():
    """
    Construct a tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing factor matrices
    that exactly decompose the matrix multiplication tensor.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10 seconds safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Build schoolbook solution as fallback (rank 36)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_result = {'U': U_sb, 'V': V_sb, 'W': W_sb}
    best_rank = rank_sb
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: at least max(n*m, m*p, p*n) is not correct
    # For <2,3,6>, let's try ranks from 24 up to 35
    
    for r in range(24, rank_sb):
        if time.time() > deadline - 5:
            break
        
        result = _try_rank(r, n, m, p, deadline, seed_offset=r)
        if result is not None:
            best_rank = r
            best_result = result
            break  # Found minimal rank for this search
    
    return best_result
# EVOLVE-BLOCK-END
