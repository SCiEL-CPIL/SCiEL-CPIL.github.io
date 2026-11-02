import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    R = U.shape[1]
    if R == 0:
        return False
        
    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                if abs(s - 1.0) > tol:
                    return False
    
    # Check zero entries
    # We need to check that for any (a,b,c) that is NOT a valid (i,j,k) triple, the sum is 0.
    # Valid triples: a = i*m+j, b = j*p+k, c = k*n+i.
    # We sample a subset of invalid triples.
    import random
    rng = random.Random(42)
    checked = 0
    max_checks = 5000
    while checked < max_checks:
        i = rng.randint(0, n - 1)
        j = rng.randint(0, m - 1)
        k = rng.randint(0, p - 1)
        a = i * m + j
        
        # Generate a random b, c
        j2 = rng.randint(0, m - 1)
        k2 = rng.randint(0, p - 1)
        b = j2 * p + k2
        
        i2 = rng.randint(0, n - 1)
        k3 = rng.randint(0, p - 1)
        c = k3 * n + i2
        
        # Skip if it's a valid entry
        if j2 == j and i2 == i and k3 == k:
            continue
            
        s = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
        if abs(s) > tol:
            return False
        checked += 1
        
    return True

def _build_schoolbook(n, m, p):
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Schoolbook baseline
    U, V, W = _build_schoolbook(n, m, p)
    
    # Verify schoolbook (should always pass)
    if _verify(U, V, W, n, m, p, tol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to empty/zero if schoolbook fails (should not happen)
        return {"U": [[0]], "V": [[0]], "W": [[0]]}
