"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time

def _verify(U, V, W, n=3, m=3, p=3):
    """Verify exact decomposition: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    rank = len(U[0]) if U and U[0] else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_3x3():
    """Rank-27 schoolbook decomposition."""
    n, m, p = 3, 3, 3
    rank = 27
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_decomposition(rank, deadline):
    """Try to find a valid rank-`rank` decomposition via iterative refinement."""
    import random
    random.seed(rank)
    
    # Possible half-integer values
    vals = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    for attempt in range(200):
        if time.time() > deadline:
            return None
        
        U = [[random.choice(vals) for _ in range(rank)] for _ in range(9)]
        V = [[random.choice(vals) for _ in range(rank)] for _ in range(9)]
        W = [[random.choice(vals) for _ in range(rank)] for _ in range(9)]
        
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    return None

def construct():
    start = time.time()
    deadline = start + 140
    
    # Try lower ranks first
    for rank in range(19, 27):
        if time.time() > deadline - 5:
            break
        result = _try_rank_decomposition(rank, deadline)
        if result is not None:
            return result
    
    return _schoolbook_3x3()
# EVOLVE-BLOCK-END
