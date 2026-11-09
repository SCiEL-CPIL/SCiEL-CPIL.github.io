import time
import numpy as np
import sys

DIMENSIONS = [3, 4, 7]

def _to_lists(mat):
    """Convert numpy array to list of lists of int/float (half-integers)."""
    return [[float(v) for v in row] for row in mat]

def _verify_tensor_fast(U, V, W, n, m, p):
    """Fast verification using tensor operations.
    Returns True if the tensor identity holds exactly.
    """
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # where a = i*m+j, b = j2*p+k, c = k2*n+i2
    # We need T[i*m+j, j*p+k, k*n+i] = 1 (diagonal)
    # and T[i*m+j, j2*p+k, k2*n+i2] = 0 for off-diagonal
    
    # Efficient check: for each r, the contribution is outer product
    # Let's build T as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    # Check diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i*m + j, j*p + k, k*n + i] - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i2 == i and j2 == j and k2 == k:
                                continue
                            if abs(T[i*m + j, j2*p + k, k2*n + i2]) > 1e-9:
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

def _try_random_search(n, m, p, target_rank, seed, max_iter=30):
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
        if _verify_tensor_fast(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _kronecker_decomposition(n, m, p):
    """
    Try to find a decomposition using Kronecker product structure.
    
    For matrix multiplication, we can write:
    C = A * B where A is n×m, B is m×p
    
    If we can find factors such that the multiplication tensor decomposes
    using Kronecker products, we may get lower rank.
    
    Strategy: Try to express the multiplication as a sum of rank-1 terms
    that have block structure.
    """
    # For now, this is a placeholder. In practice, we'd need to find
    # specific Kronecker decompositions.
    # Return None to indicate no improvement found.
    return None

def _structured_search(n, m, p, deadline):
    """
    Structured search for low-rank decompositions.
    
    For n=3, m=4, p=7:
    - Schoolbook rank = 84
    - Lower bound = max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    
    Strategy:
    1. Try random searches at various ranks (from low to high)
    2. Try structured block decompositions
    """
    # Start with schoolbook
    U_sb, V_sb, W_sb, sb_rank = _schoolbook(n, m, p)
    
    # Try random search at various ranks (from low to high)
    # Be conservative: try ranks that might be achievable
    target_ranks = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 15:  # Safety margin
            break
        
        # Try multiple seeds
        for seed in range(15):
            if time.time() > deadline - 15:
                break
            
            result = _try_random_search(n, m, p, target_rank, seed, max_iter=25)
            if result is not None:
                U, V, W = result
                return U, V, W, target_rank
    
    # If nothing found, return schoolbook
    return U_sb, V_sb, W_sb, sb_rank

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    U, V, W, rank = _structured_search(n, m, p, deadline)
    
    # Convert to lists
    result = {
        "U": _to_lists(U),
        "V": _to_lists(V),
        "W": _to_lists(W)
    }
    
    return result
