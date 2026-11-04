import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 4]

def _verify_fast(U, V, W, n, m, p):
    """Fast vectorized verification of tensor decomposition."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T[u,v,w] = 1 if (u,v,w) corresponds to valid (i,j,k)
    # u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Compute the tensor from U, V, W: T_comp[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
    # This is a 3-way outer product sum
    # Efficient: T_comp = einsum('ur,vr,wr->uvw', U, V, W)
    T_comp = np.einsum('ur,vr,wr->uvw', U, V, W, optimize=True)
    
    return np.allclose(T_comp, T, atol=1e-9, rtol=0)

def _verify_sparse(U, V, W, n, m, p):
    """Sparse verification - only check active and a sample of inactive entries."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    nm, mp, pn = n * m, m * p, p * n
    
    # Check active entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                val = np.dot(U[u, :] * V[v, :], W[w, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check a random sample of inactive entries
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                valid.add((u, v, w))
    
    rng = np.random.RandomState(42)
    n_inactive = nm * mp * pn - len(valid)
    n_check = min(500, n_inactive)
    
    for _ in range(n_check):
        u = rng.randint(nm)
        v = rng.randint(mp)
        w = rng.randint(pn)
        if (u, v, w) not in valid:
            val = np.dot(U[u, :] * V[v, :], W[w, :])
            if abs(val) > 1e-9:
                return False
    
    return True

def _schoolbook(n, m, p):
    """Standard O(nmp) multiplication."""
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

def _winograd_style(n, m, p, deadline):
    """
    Try to construct a decomposition using Winograd-style techniques.
    For small matrices, we can use explicit known decompositions.
    """
    # For 3x3 * 3x4, let's try to use a structured approach
    # The key insight: we can factor the multiplication using intermediate transforms
    
    # One approach: use the fact that (3x3)*(3x4) = (3x3)*(3x3) + (3x3)*(3x1)
    # But this doesn't directly help with rank reduction
    
    # Alternative: try to find a decomposition where some entries share factors
    # This is essentially what Strassen's algorithm does for 2x2
    
    # For 3x3 * 3x4, let's try a direct structured approach
    # We'll attempt to use a smaller number of multiplications by finding
    # a basis where the tensor has lower rank
    
    # Let's try to construct a decomposition with rank < 36
    # by exploiting the structure
    
    # For now, return None and let other methods handle it
    return None

def _iterative_improvement(U, V, W, n, m, p, deadline):
    """
    Try to iteratively improve the decomposition by:
    1. Checking if columns can be removed
    2. Trying to merge similar columns
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try to remove each column
        for c in range(rank):
            if time.time() > deadline - 10:
                return U, V, W
            
            U_test = np.delete(U, c, axis=1)
            V_test = np.delete(V, c, axis=1)
            W_test = np.delete(W, c, axis=1)
            
            if _verify_sparse(U_test, V_test, W_test, n, m, p):
                U, V, W = U_test, V_test, W_test
                rank -= 1
                improved = True
                break
        
        # Try to find linear dependencies and reduce rank
        if rank > 1 and time.time() < deadline - 10:
            # Check if any column is a linear combination of others
            # This is complex, so we'll skip for now
            pass
    
    return U, V, W

def _structured_search(n, m, p, deadline):
    """
    Try structured decompositions based on known results.
    For 3x3 * 3x4, we know the schoolbook rank is 36.
    We'll try to find a lower rank decomposition.
    """
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # Upper bound: 36 (schoolbook)
    
    # Let's try ranks from 20 to 35
    # Use a smarter search: start with schoolbook and try to modify it
    
    # First, get the schoolbook decomposition
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Try to reduce rank by finding redundant columns
    U, V, W = _iterative_improvement(U_sb, V_sb, W_sb, n, m, p, deadline)
    
    if U.shape[1] < U_sb.shape[1]:
        return U, V, W
    
    return U_sb, V_sb, W_sb

def _try_rank_r(n, m, p, r, deadline, max_attempts=200):
    """
    Try to find a rank-r decomposition using a smarter search.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
            v = j * p + k
            w = k * n + i
            T[u, v, w] = 1.0
    
    # Try random initialization with structured patterns
    rng = np.random.RandomState(12345)
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 15:
            break
        
        # Use a mix of random and structured initialization
        if attempt % 3 == 0:
            # Pure random
            U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(nm, r))
            V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(mp, r))
            W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(pn, r))
        elif attempt % 3 == 1:
            # Structured: start with schoolbook pattern and modify
            U, V, W = _schoolbook(n, m, p)
            # Randomly zero out some entries
            for _ in range(r // 2):
                col = rng.randint(r)
                U[rng.randint(nm), col] = 0
                V[rng.randint(mp), col] = 0
                W[rng.randint(pn), col] = 0
            # Scale down to rank r
            if r < U.shape[1]:
                U = U[:, :r]
                V = V[:, :r]
                W = W[:, :r]
        else:
            # Hybrid: use a subset of schoolbook columns
            U, V, W = _schoolbook(n, m, p)
            cols = rng.choice(U.shape[1], size=r, replace=False)
            U = U[:, cols]
            V = V[:, cols]
            W = W[:, cols]
            # Add some noise
            U += rng.normal(0, 0.1, size=U.shape)
            V += rng.normal(0, 0.1, size=V.shape)
            W += rng.normal(0, 0.1, size=W.shape)
            # Round to half-integers
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
        
        if _verify_sparse(U, V, W, n, m, p):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try iterative improvement
    U_imp, V_imp, W_imp = _iterative_improvement(U, V, W, n, m, p, deadline - 30)
    if U_imp.shape[1] < best_rank:
        best_rank = U_imp.shape[1]
        best_U, best_V, best_W = U_imp, V_imp, W_imp
    
    # Try structured search for lower ranks
    for r in range(best_rank - 1, max(15, best_rank - 15), -1):
        if time.time() > deadline - 20:
            break
        
        result = _try_rank_r(n, m, p, r, deadline - 10, max_attempts=100)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_fast(U_r, V_r, W_r, n, m, p):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Return the best found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
