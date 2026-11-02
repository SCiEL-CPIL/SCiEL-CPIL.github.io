"""Search for a low-rank exact tensor decomposition of (2,2,8) matrix multiplication."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 2, 8]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16
SCHOOBOOK_RANK = N * M * P  # 32

def verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W form an exact decomposition. Returns True/False."""
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    # Build a set of valid (a,b,c) indices
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    # Check all entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                expected = 1 if (a, b, c) in valid else 0
                if s != expected:
                    return False
    return True

def schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    n, m, p = N, M, P
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

def strassen_decomposition_228():
    """
    Construct a rank-28 decomposition for M_{2,2,8} using 4 blocks of Strassen M_{2,2,2}.
    """
    n, m, p = N, M, P
    # Correct Strassen terms for M_{2,2,2}
    # u, v, w are vectors of length 4 (flattened 2x2 matrices: [00, 01, 10, 11])
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    # 4 blocks of 2x2 in the 2x8 matrix
    # B is 2x8. Split into 4 blocks of 2x2: columns (0,1), (2,3), (4,5), (6,7)
    # C is 2x8. Split into 4 blocks of 2x2: columns (0,1), (2,3), (4,5), (6,7)
    
    for block_idx in range(4):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = terms[term_idx]
            
            # Fill U
            # U index is i (0..3) corresponding to A_{i//2, i%2}
            for i in range(4):
                U[i][col] = u[i]
                
            # Fill V
            # v is indexed by local (j, k) where j in 0..1, k in 0..1
            # Global B index: b = j * p + k_global
            # k_global = 2 * block_idx + k_local
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    b_idx = j * P + k_global
                    v_idx = j * 2 + k_local
                    V[b_idx][col] = v[v_idx]
                    
            # Fill W
            # w is indexed by local (i, k) where i in 0..1, k in 0..1
            # Global C index: c = k_global * n + i
            # k_global = 2 * block_idx + k_local
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    c_idx = k_global * N + i
                    w_idx = i * 2 + k_local
                    W[c_idx][col] = w[w_idx]
                    
    return U, V, W

def construct():
    """Entry point: return the best decomposition found."""
    deadline = time.time() + 120
    best = None
    try:
        # Try rank 28 first
        U, V, W = strassen_decomposition_228()
        if verify(U, V, W):
            best = {"U": U, "V": V, "W": W}
    except Exception:
        pass
    if best is not None:
        return best
    # Fallback to schoolbook
    try:
        U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}
    except Exception:
        # Last resort: return empty (should not happen)
        return {"U": [[0]*32 for _ in range(4)], "V": [[0]*32 for _ in range(16)], "W": [[0]*32 for _ in range(16)]}
