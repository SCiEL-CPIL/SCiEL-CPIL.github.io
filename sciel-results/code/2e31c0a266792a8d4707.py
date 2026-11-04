"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, n*p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    return T

def _check_decomposition(U, V, W):
    """Check if U, V, W exactly decompose the tensor."""
    T = _build_tensor()
    # Compute sum_c U[:,c] * V[:,c] * W[:,c]
    r = U.shape[1]
    result = np.zeros_like(T)
    for c in range(r):
        result += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
    return np.array_equal(result, T)

def _random_init(n, m, p, rank, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    U = rng.randint(-2, 3, size=(n*m, rank))
    V = rng.randint(-2, 3, size=(m*p, rank))
    W = rng.randint(-2, 3, size=(p*n, rank))
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s safety margin
    
    # Start with schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    # Try to find lower rank using randomized search
    T = _build_tensor()
    
    # Try ranks from a lower bound up to schoolbook
    # Lower bound: at least max(n*m, m*p, n*p) / something... 
    # Actually, a simple lower bound is the tensor rank lower bound
    # For now, try from 20 up to schoolbook
    
    for target_rank in range(20, rank_school):
        if time.time() > deadline:
            break
        
        # Try multiple random inits
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            U_r, V_r, W_r = _random_init(n, m, p, target_rank, seed=None)
            
            # Check if this random init happens to work (very unlikely)
            if _check_decomposition(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r.tolist(), V_r.tolist(), W_r.tolist()
                best_rank = target_rank
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
