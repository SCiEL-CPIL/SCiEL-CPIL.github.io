import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=2, m=3, p=9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = n, m, p
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val) > 1e-9:
                    return False
                # Check that the tensor entry is 1
                # The tensor entry T[a,b,c] should be 1
                # But we're checking that the decomposition gives the correct value
                # Actually, we need to check that the sum equals the tensor entry
    # Better: check all entries
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                val = np.dot(U[a], V[b] * W[c])
                # Determine what the tensor entry should be
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if j == j2 and k == k2 and i == i2:
                    expected = 1
                else:
                    expected = 0
                if abs(val - expected) > 1e-9:
                    return False
    return True

def _schoolbook(n=2, m=3, p=9):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _search_decomposition(n=2, m=3, p=9, max_rank=54, min_rank=30, time_limit=120):
    """Search for a low-rank decomposition."""
    t0 = time.time()
    best_rank = max_rank
    best_result = None
    
    for rank in range(min_rank, max_rank + 1):
        if time.time() - t0 > time_limit - 5:
            break
        
        # Try multiple random restarts for this rank
        n_restarts = 100
        for restart in range(n_restarts):
            if time.time() - t0 > time_limit - 5:
                break
            
            # Random initialization with small integers
            rng = np.random.default_rng(rank * 1000 + restart)
            U = rng.integers(-2, 3, size=(n * m, rank))
            V = rng.integers(-2, 3, size=(m * p, rank))
            W = rng.integers(-2, 3, size=(p * n, rank))
            
            # Check if this random initialization happens to work (very unlikely)
            if _verify(U, V, W, n, m, p):
                best_rank = rank
                best_result = (U, V, W)
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_rank, best_result
    else:
        # Return schoolbook
        U, V, W = _schoolbook(n, m, p)
        return max_rank, (U, V, W)

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition
    try:
        rank, (U, V, W) = _search_decomposition(n, m, p, max_rank=54, min_rank=30, time_limit=140)
        U_list = [[int(x) for x in row] for row in U.tolist()]
        V_list = [[int(x) for x in row] for row in V.tolist()]
        W_list = [[int(x) for x in row] for row in W.tolist()]
        return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
