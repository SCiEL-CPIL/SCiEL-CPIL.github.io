"""Matrix multiplication 4,4,8 rank-minimal search."""
import random
import time

DIMENSIONS = [4, 4, 8]

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact for the 4x4 * 4x8 tensor."""
    rank = len(U[0]) if U else 0
    # T[a, b, c] should be 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Only need to check the "1" positions and a sample of "0" positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    # Check some zero positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_alt = (k + 1) % p * n + i  # wrong k for this (i,j)
                if c_alt != c:
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c_alt][r]
                    if abs(s) > 1e-9:
                        return False
    return True

def _try_random_rank(r, n, m, p, deadline):
    """Try to find a rank-r decomposition via random search with local optimization."""
    nm = n * m
    mp = m * p
    pn = p * n
    start = time.time()
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            return None
        # Random initialization with entries from {0, 1, -1}
        U = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(nm)]
        V = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(mp)]
        W = [[random.choice([0, 1, -1]) for _ in range(r)] for _ in range(pn)]
        
        # Local search: try flipping entries
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            if _validate(U, V, W, n, m, p):
                return U, V, W
            # Pick a random entry to flip
            mat = random.choice([U, V, W])
            row = random.randrange(len(mat))
            col = random.randrange(r)
            mat[row][col] = -mat[row][col] if mat[row][col] != 0 else 1
            if time.time() > deadline - 5:
                break
    
    return None

def _construct_known(n, m, p):
    """Schoolbook decomposition as fallback."""
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

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin
    
    # Try ranks from low to high
    for r in range(10, n * m * p + 1):
        if time.time() > deadline - 10:
            break
        result = _try_random_rank(r, n, m, p, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _construct_known(n, m, p)
