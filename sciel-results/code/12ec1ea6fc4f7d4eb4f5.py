"""Matrix multiplication 3,4,5 - rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 5]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) valid."""
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all (i,j,k)."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    # Check each (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank(n, m, p, rank, time_budget=10.0):
    """Try to find a decomposition of given rank. Returns (U,V,W) or None."""
    start = time.time()
    U_shape = (n * m, rank)
    V_shape = (m * p, rank)
    W_shape = (p * n, rank)
    
    # Initialize with schoolbook and try to zero out entries
    U = np.zeros(U_shape, dtype=np.float64)
    V = np.zeros(V_shape, dtype=np.float64)
    W = np.zeros(W_shape, dtype=np.float64)
    
    # Start from schoolbook
    for c in range(n * m * p):
        i = c // (m * p)
        j = (c // p) % m
        k = c % p
        U[i * m + j, c] = 1.0
        V[j * p + k, c] = 1.0
        W[k * n + i, c] = 1.0
    
    # Try to reduce rank by setting some entries to zero and checking if we can fix
    # Strategy: try to merge columns by setting some U,V,W entries to zero
    # and compensating with other entries
    
    # For now, try a simple approach: start from schoolbook at target rank
    # by randomly permuting and combining
    
    # Use random restarts with iterative improvement
    np.random.seed(42)
    
    for attempt in range(200):
        if time.time() - start > time_budget:
            break
        
        U = np.zeros(U_shape, dtype=np.float64)
        V = np.zeros(V_shape, dtype=np.float64)
        W = np.zeros(W_shape, dtype=np.float64)
        
        # Random initialization with some structure
        for c in range(rank):
            # Pick random (i,j) for U, (j',k) for V, (k',i') for W
            # Try to create valid triples
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        
        # Check if this works (it won't generally)
        # Try iterative repair
        for it in range(500):
            if time.time() - start > time_budget:
                break
            
            # Check which (i,j,k) are satisfied
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        if abs(s - 1.0) > 1e-9:
                            errors.append((i, j, k, s))
            
            if not errors:
                return U.tolist(), V.tolist(), W.tolist()
            
            # Pick a random error and try to fix it
            i, j, k, s = errors[np.random.randint(len(errors))]
            
            # Find a column c where we can adjust
            c = np.random.randint(rank)
            
            # Try setting U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
            # But this might break other constraints
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_fallback = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fallback = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fallback = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fallback[i * m + j][column] = 1
                V_fallback[j * p + k][column] = 1
                W_fallback[k * n + i][column] = 1
                column += 1
    
    # Try to find lower rank
    # Known lower bounds: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # Upper bound from schoolbook: 60
    # Try ranks from 20 up
    
    deadline = time.time() + 120.0
    best_result = {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    
    for rank in range(20, schoolbook_rank + 1):
        if time.time() > deadline - 5.0:
            break
        
        time_budget = min(10.0, max(1.0, deadline - time.time() - 5.0))
        result = _try_rank(n, m, p, rank, time_budget)
        
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_decomposition(U, V, W, n, m, p, rank):
                best_result = {"U": U, "V": V, "W": W}
                break
    
    return best_result
