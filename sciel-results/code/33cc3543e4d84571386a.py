"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

import time
import math
import functools
import numpy as np

# EVOLVE-BLOCK-START

def _base_strassen_222():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    U has shape (4, 7)  [n*m = 2*2]
    V has shape (4, 7)  [m*p = 2*2]
    W has shape (4, 7)  [p*n = 2*2]
    """
    n = m = p = 2
    rank = 7
    
    U = [
        [1, 0, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0],
    ]
    
    V = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
    ]
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    return U, V, W

def _schoolbook_decomp(n, m, p):
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

def _combine_split_m(n, m1, m2, p, U1, V1, W1, U2, V2, W2):
    """
    Combine decompositions for (n, m1, p) and (n, m2, p) into (n, m1+m2, p).
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    total_rank = r1 + r2
    m = m1 + m2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill U
    for i in range(n):
        for j in range(m1):
            U[i * m + j] = U1[i * m1 + j] + [0] * r2
        for j in range(m2):
            U[i * m + (m1 + j)] = [0] * r1 + U2[i * m2 + j]
            
    # Fill V
    for j in range(m1):
        for k in range(p):
            V[j * p + k] = V1[j * p + k] + [0] * r2
    for j in range(m2):
        for k in range(p):
            V[(m1 + j) * p + k] = [0] * r1 + V2[j * p + k]
            
    # Fill W
    for k in range(p):
        for i in range(n):
            W[k * n + i] = [x + y for x, y in zip(W1[k * n + i], W2[k * n + i])]
            
    return U, V, W

def _combine_split_n(n1, n2, m, p, U1, V1, W1, U2, V2, W2):
    """
    Combine decompositions for (n1, m, p) and (n2, m, p) into (n1+n2, m, p).
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    total_rank = r1 + r2
    n = n1 + n2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill U
    for i in range(n1):
        for j in range(m):
            U[i * m + j] = U1[i * m + j] + [0] * r2
    for i in range(n2):
        for j in range(m):
            U[(n1 + i) * m + j] = [0] * r1 + U2[i * m + j]
            
    # Fill V
    # V is shared, so we just concatenate columns
    for j in range(m):
        for k in range(p):
            V[j * p + k] = V1[j * p + k] + [0] * r2
            # Wait, V2 is also m*p x r2. We need to add it to the second part?
            # No, the rank dimension is concatenated.
            # V[:, 0:r1] = V1
            # V[:, r1:r1+r2] = V2
            # So V[j*p+k] = V1[j*p+k] + V2[j*p+k] is WRONG.
            # It should be:
            # V[j*p+k][0:r1] = V1[j*p+k]
            # V[j*p+k][r1:r1+r2] = V2[j*p+k]
            # My previous line was: V[j * p + k] = V1[j * p + k] + [0] * r2
            # This sets the first r1 entries to V1 and rest to 0.
            # I need to also set the second part to V2.
            V[j * p + k] = V1[j * p + k] + V2[j * p + k] # This adds them, which is wrong if they are in different rank blocks.
            # Correction:
            V[j * p + k] = V1[j * p + k] + [0] * r2
            # And then we need to add V2 to the second block?
            # No, the decomposition is a sum of terms.
            # Term 1: U1 V1 W1. Term 2: U2 V2 W2.
            # We map Term 1 to rank indices 0..r1-1 and Term 2 to r1..r1+r2-1.
            # So V should have V1 in the first r1 columns and V2 in the next r2 columns.
            # So V[j*p+k] = V1[j*p+k] + V2[j*p+k] is incorrect because V1 and V2 have different widths.
            # We need to construct the row manually.
            
    # Let's redo V construction
    for j in range(m):
        for k in range(p):
            row = [0] * total_rank
            for c in range(r1):
                row[c] = V1[j * p + k][c]
            for c in range(r2):
                row[r1 + c] = V2[j * p + k][c]
            V[j * p + k] = row
            
    # Fill W
    # W is indexed by (k, i).
    # For i < n1, we use W1. For i >= n1, we use W2.
    # Rank indices 0..r1-1 correspond to W1. Rank indices r1..r1+r2-1 correspond to W2.
    # So for i < n1:
    #   W[k*n+i][0:r1] = W1[k*n1+i]
    #   W[k*n+i][r1:r1+r2] = 0
    # For i >= n1:
    #   W[k*n+i][0:r1] = 0
    #   W[k*n+i][r1:r1+r2] = W2[k*n2+(i-n1)]
    
    for k in range(p):
        for i in range(n1):
            row = [0] * total_rank
            for c in range(r1):
                row[c] = W1[k * n1 + i][c]
            W[k * n + i] = row
            
        for i in range(n2):
            row = [0] * total_rank
            for c in range(r2):
                row[r1 + c] = W2[k * n2 + i][c]
            W[k * n + (n1 + i)] = row
            
    return U, V, W

def _combine_split_p(n, m, p1, p2, U1, V1, W1, U2, V2, W2):
    """
    Combine decompositions for (n, m, p1) and (n, m, p2) into (n, m, p1+p2).
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    total_rank = r1 + r2
    p = p1 + p2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill U
    # U is shared. Concatenate columns.
    for i in range(n):
        for j in range(m):
            row = [0] * total_rank
            for c in range(r1):
                row[c] = U1[i * m + j][c]
            for c in range(r2):
                row[r1 + c] = U2[i * m + j][c]
            U[i * m + j] = row
            
    # Fill V
    # V is indexed by (j, k).
    # For k < p1, use V1. For k >= p1, use V2.
    # Rank indices 0..r1-1 for V1, r1..r1+r2-1 for V2.
    for j in range(m):
        for k in range(p1):
            row = [0] * total_rank
            for c in range(r1):
                row[c] = V1[j * p1 + k][c]
            V[j * p + k] = row
            
        for k in range(p2):
            row = [0] * total_rank
            for c in range(r2):
                row[r1 + c] = V2[j * p2 + k][c]
            V[j * p + (p1 + k)] = row
            
    # Fill W
    # W is indexed by (k, i).
    # For k < p1, use W1. For k >= p1, use W2.
    for k in range(p1):
        for i in range(n):
            row = [0] * total_rank
            for c in range(r1):
                row[c] = W1[k * n + i][c]
            W[k * n + i] = row
            
        for k in range(p2):
            for i in range(n):
                row = [0] * total_rank
                for c in range(r2):
                    row[r1 + c] = W2[k * n + i][c]
                W[(p1 + k) * n + i] = row
                
    return U, V, W

@functools.lru_cache(maxsize=None)
def _solve(n, m, p):
    # Base cases
    if n == 1 and m == 1 and p == 1:
        return [[1]], [[1]], [[1]]
    
    if m == 1:
        return _schoolbook_decomp(n, 1, p)
    if n == 1:
        return _schoolbook_decomp(1, m, p)
    if p == 1:
        return _schoolbook_decomp(n, m, 1)
    
    if n == 2 and m == 2 and p == 2:
        return _base_strassen_222()
    
    best_rank = float('inf')
    best_result = None
    
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1 = _solve(n1, m, p)
        U2, V2, W2 = _solve(n2, m, p)
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = r1 + r2
        if total_rank < best_rank:
            U, V, W = _combine_split_n(n1, n2, m, p, U1, V1, W1, U2, V2, W2)
            best_rank = total_rank
            best_result = (U, V, W)
            
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = _solve(n, m1, p)
        U2, V2, W2 = _solve(n, m2, p)
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = r1 + r2
        if total_rank < best_rank:
            U, V, W = _combine_split_m(n, m1, m2, p, U1, V1, W1, U2, V2, W2)
            best_rank = total_rank
            best_result = (U, V, W)
            
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1 = _solve(n, m, p1)
        U2, V2, W2 = _solve(n, m, p2)
        r1 = len(U1[0])
        r2 = len(U2[0])
        total_rank = r1 + r2
        if total_rank < best_rank:
            U, V, W = _combine_split_p(n, m, p1, p2, U1, V1, W1, U2, V2, W2)
            best_rank = total_rank
            best_result = (U, V, W)
            
    if best_result is None:
        # Fallback to schoolbook
        return _schoolbook_decomp(n, m, p)
        
    return best_result

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    try:
        U, V, W = _solve(n, m, p)
        
        if time.time() < deadline - 5:
            if not _verify_decomposition(U, V, W, n, m, p):
                # Fallback
                U, V, W = _schoolbook_decomp(n, m, p)
                
        return {"U": U, "V": V, "W": W}
    except Exception:
        U, V, W = _schoolbook_decomp(n, m, p)
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
