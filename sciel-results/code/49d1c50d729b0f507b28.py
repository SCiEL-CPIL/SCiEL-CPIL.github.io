"""Search for a low-rank exact tensor decomposition of (2,2,8) matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 8]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16

def verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W form an exact decomposition. Returns True/False."""
    nm, mp, pn = n*m, m*p, p*n
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
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
    
    for block_idx in range(4):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = terms[term_idx]
            
            for i in range(4):
                U[i][col] = u[i]
                
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    b_idx = j * P + k_global
                    v_idx = j * 2 + k_local
                    V[b_idx][col] = v[v_idx]
                    
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * block_idx + k_local
                    c_idx = k_global * N + i
                    w_idx = i * 2 + k_local
                    W[c_idx][col] = w[w_idx]
                    
    return U, V, W

def construct():
    """Entry point: return the best decomposition found."""
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # First try the known rank-28 Strassen decomposition
    try:
        U, V, W = strassen_decomposition_228()
        if verify(U, V, W):
            # Optional: Check for linear dependencies to reduce rank
            # This is a heuristic and might not always succeed in constructing a valid lower rank decomposition
            # but it's worth trying if we have time.
            if time.time() < deadline - 10:
                try:
                    U_red, V_red, W_red = reduce_rank_strassen(U, V, W)
                    if U_red and verify(U_red, V_red, W_red):
                        return {"U": U_red, "V": V_red, "W": W_red}
                except Exception:
                    pass
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}

def reduce_rank_strassen(U, V, W):
    """
    Attempt to reduce the rank of the Strassen decomposition by finding linear dependencies.
    """
    import numpy as np
    rank = len(U[0])
    nm, mp, pn = NM, MP, PN
    total_entries = nm * mp * pn
    
    A = np.zeros((total_entries, rank), dtype=float)
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                entry = a * mp * pn + b * pn + c
                for r in range(rank):
                    A[entry, r] = U[a][r] * V[b][r] * W[c][r]
    
    U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
    tol = 1e-10
    numerical_rank = np.sum(s > tol)
    
    if numerical_rank < rank:
        # We have dependencies.
        # Simple strategy: remove the last column if it's dependent on the others.
        # This is not guaranteed to work, but we can try.
        # A more robust method would be to find a basis for the column space.
        # For now, we just return the original if we can't easily construct the reduced one.
        pass
        
    return U, V, W
