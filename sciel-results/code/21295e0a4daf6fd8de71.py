import time
import numpy as np
import sys

DIMENSIONS = [3, 4, 7]

def _to_lists(mat):
    """Convert numpy array to list of lists of int/float (half-integers)."""
    return [[float(v) for v in row] for row in mat]

def _verify_tensor(U, V, W, n, m, p):
    """Verify that the tensor identity holds:
    sum_r U[i*m+j, r] * V[j2*p+k, r] * W[k2*n+i2, r] = 1 if (i,j,k)==(i2,j2,k2) else 0
    """
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # Build the full tensor check
    # T[(i,j), (j2,k), (k2,i2)] = sum_r U[i*m+j, r] * V[j2*p+k, r] * W[k2*n+i2, r]
    # Should equal 1 if i==i2, j==j2, k==k2, else 0
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Diagonal: should be 1
                s = 0.0
                for r in range(rank):
                    s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
                
                # Off-diagonal: should be 0
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i2 == i and j2 == j and k2 == k:
                                continue
                            s = 0.0
                            for r in range(rank):
                                s += U[i*m + j, r] * V[j2*p + k, r] * W[k2*n + i2, r]
                            if abs(s) > 1e-9:
                                return False
    return True

def _schoolbook(n, m, p):
    """Return the standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W, rank

def _try_random_search(n, m, p, target_rank, seed, max_iter=50):
    """Try to find a rank-target_rank decomposition via random search."""
    np.random.seed(seed)
    
    for iteration in range(max_iter):
        # Generate random integer matrices with small entries
        U = np.random.randint(-3, 4, size=(n*m, target_rank)).astype(float)
        V = np.random.randint(-3, 4, size=(m*p, target_rank)).astype(float)
        W = np.random.randint(-3, 4, size=(p*n, target_rank)).astype(float)
        
        # Quick spot check before full verification
        ok = True
        # Check a few diagonal entries
        for _ in range(5):
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
            if abs(s - 1.0) > 1e-6:
                ok = False
                break
        
        if not ok:
            continue
        
        # Check a few off-diagonal entries
        for _ in range(5):
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            k = np.random.randint(0, p)
            i2 = np.random.randint(0, n)
            j2 = np.random.randint(0, m)
            k2 = np.random.randint(0, p)
            if i2 == i and j2 == j and k2 == k:
                continue
            s = np.dot(U[i*m+j, :], V[j2*p+k, :] * W[k2*n+i2, :])
            if abs(s) > 1e-6:
                ok = False
                break
        
        if not ok:
            continue
        
        # Full verification
        if _verify_tensor(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _structured_search(n, m, p, deadline):
    """
    Structured search for low-rank decompositions.
    
    Key insight: The matrix multiplication tensor M_{n,m,p} has a specific structure.
    We can try to exploit block structure.
    
    For n=3, m=4, p=7:
    - Schoolbook rank = 84
    - Lower bound = max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    
    Strategy:
    1. Start from schoolbook and try to merge terms
    2. Try random searches at various ranks
    3. Try structured block decompositions
    """
    n, m, p = 3, 4, 7
    
    # Start with schoolbook
    U_sb, V_sb, W_sb, sb_rank = _schoolbook(n, m, p)
    
    # Try random search at various ranks (from low to high)
    # Be conservative: try ranks that might be achievable
    target_ranks = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:  # Safety margin
            break
        
        # Try multiple seeds
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            
            result = _try_random_search(n, m, p, target_rank, seed, max_iter=20)
            if result is not None:
                U, V, W = result
                return U, V, W, target_rank
    
    # If nothing found, return schoolbook
    return U_sb, V_sb, W_sb, sb_rank

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    U, V, W, rank = _structured_search(n, m, p, deadline)
    
    # Convert to lists
    result = {
        "U": _to_lists(U),
        "V": _to_lists(V),
        "W": _to_lists(W)
    }
    
    return result
