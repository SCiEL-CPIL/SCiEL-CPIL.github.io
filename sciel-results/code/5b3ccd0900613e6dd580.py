import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_terms():
    """Return Strassen's 7 rank-1 terms for 2x2x2 matrix multiplication.
    Each term is (U_dict, V_dict, W_dict) where:
      U_dict maps (i,j) -> value
      V_dict maps (j,k) -> value  
      W_dict maps (k,i) -> value
    """
    terms = []
    # M1 = (a00+a11)(b00+b11) -> C00:+1, C11:+1
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    # M2 = (a10+a11)b00 -> C10:+1, C11:-1
    terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}))
    # M3 = a00(b01-b11) -> C01:+1, C11:+1
    terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}))
    # M4 = a11(b10-b00) -> C00:+1, C10:+1
    terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}))
    # M5 = (a00+a01)b11 -> C00:-1, C01:+1
    terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}))
    # M6 = (a10-a00)(b00+b01) -> C11:+1
    terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
    # M7 = (a01-a11)(b10+b11) -> C00:+1
    terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}))
    return terms

def _schoolbook_terms(n, m, p):
    """Return schoolbook rank-1 terms for (n,m,p) matrix multiplication."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(({(i,j):1}, {(j,k):1}, {(k,i):1}))
    return terms

def _decompose_recursive(n, m, p, deadline):
    """Recursively decompose (n,m,p) matrix multiplication into rank-1 terms.
    Returns list of (U_dict, V_dict, W_dict) tuples with local indices.
    """
    if time.time() > deadline:
        return None
    
    # Base case: 2x2x2 uses Strassen
    if n == 2 and m == 2 and p == 2:
        return _strassen_terms()
    
    # If any dimension is 1, use schoolbook
    if n == 1 or m == 1 or p == 1:
        return _schoolbook_terms(n, m, p)
    
    # Try splitting along m dimension: C = A[:,:m1]@B[:m1,:] + A[:,m1:]@B[m1:,:]
    # Sub-problems: (n, m1, p) and (n, m2, p) where m2 = m - m1
    
    best_terms = None
    best_rank = n * m * p  # schoolbook rank
    
    # Try all possible split points for m
    for m1 in range(1, m):
        m2 = m - m1
        if time.time() > deadline:
            break
        
        terms1 = _decompose_recursive(n, m1, p, deadline)
        if terms1 is None:
            continue
        
        terms2 = _decompose_recursive(n, m2, p, deadline)
        if terms2 is None:
            continue
        
        # Combine: shift j indices in terms2 by m1
        combined = list(terms1)
        for U_d, V_d, W_d in terms2:
            new_U_d = {(i, j + m1): val for (i, j), val in U_d.items()}
            new_V_d = {(j + m1, k): val for (j, k), val in V_d.items()}
            combined.append((new_U_d, new_V_d, W_d))
        
        rank = len(combined)
        if rank < best_rank:
            best_rank = rank
            best_terms = combined
    
    # Also try splitting along n dimension: C = A[:n1,:]@B + A[n1:,:]@B
    # Sub-problems: (n1, m, p) and (n2, m, p) where n2 = n - n1
    for n1 in range(1, n):
        n2 = n - n1
        if time.time() > deadline:
            break
        
        terms1 = _decompose_recursive(n1, m, p, deadline)
        if terms1 is None:
            continue
        
        terms2 = _decompose_recursive(n2, m, p, deadline)
        if terms2 is None:
            continue
        
        # Combine: shift i indices in terms2 by n1
        combined = list(terms1)
        for U_d, V_d, W_d in terms2:
            new_U_d = {(i + n1, j): val for (i, j), val in U_d.items()}
            new_W_d = {(k, i + n1): val for (k, i), val in W_d.items()}
            combined.append((new_U_d, V_d, new_W_d))
        
        rank = len(combined)
        if rank < best_rank:
            best_rank = rank
            best_terms = combined
    
    # Also try splitting along p dimension: C = A@B[:,:p1] + A@B[:,p1:]
    # Sub-problems: (n, m, p1) and (n, m, p2) where p2 = p - p1
    for p1 in range(1, p):
        p2 = p - p1
        if time.time() > deadline:
            break
        
        terms1 = _decompose_recursive(n, m, p1, deadline)
        if terms1 is None:
            continue
        
        terms2 = _decompose_recursive(n, m, p2, deadline)
        if terms2 is None:
            continue
        
        # Combine: shift k indices in terms2 by p1
        combined = list(terms1)
        for U_d, V_d, W_d in terms2:
            new_V_d = {(j, k + p1): val for (j, k), val in V_d.items()}
            new_W_d = {(k + p1, i): val for (k, i), val in W_d.items()}
            combined.append((U_d, new_V_d, new_W_d))
        
        rank = len(combined)
        if rank < best_rank:
            best_rank = rank
            best_terms = combined
    
    if best_terms is None:
        # Fallback to schoolbook
        return _schoolbook_terms(n, m, p)
    
    return best_terms

def _build_from_terms(terms, n, m, p):
    """Convert list of (U_dict, V_dict, W_dict) terms into U, V, W matrices.
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    """
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (U_d, V_d, W_d) in enumerate(terms):
        for (i, j), val in U_d.items():
            U[i * m + j][c] = val
        for (j, k), val in V_d.items():
            V[j * p + k][c] = val
        for (k, i), val in W_d.items():
            W[k * n + i][c] = val
    
    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    """Verify the tensor identity using NumPy vectorized operations."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    idx_i, idx_j, idx_k = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    idx_i = idx_i.ravel()
    idx_j = idx_j.ravel()
    idx_k = idx_k.ravel()
    
    u_rows = idx_i * m + idx_j
    v_rows = idx_j * p + idx_k
    w_rows = idx_k * n + idx_i
    
    prod = U_np[u_rows, :] * V_np[v_rows, :] * W_np[w_rows, :]
    sums = prod.sum(axis=1)
    
    return np.allclose(sums, 1.0, atol=1e-9)

def _schoolbook_factors(n, m, p):
    """Return (U, V, W) as lists of lists for schoolbook multiplication."""
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # Try recursive decomposition
    terms = _decompose_recursive(n, m, p, deadline)
    
    if terms is not None:
        U, V, W = _build_from_terms(terms, n, m, p)
        if time.time() < deadline - 5 and _verify_fast(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
