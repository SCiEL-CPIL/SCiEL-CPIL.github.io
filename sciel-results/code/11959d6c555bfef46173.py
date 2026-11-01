"""
Search for a low-rank exact tensor decomposition of (2,2,8) matrix multiplication.
Uses block Strassen algorithm to achieve rank 28.
"""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16

def verify(U, V, W, n=N, m=M, p=P):
    """
    Verify that U, V, W form an exact decomposition.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    The tensor T(a,b,c) = sum_r U[a,r] V[b,r] W[c,r]
    Should be 1 if (a,b,c) corresponds to a valid multiplication term, 0 otherwise.
    Valid terms: A[i,j] * B[j,k] -> C[i,k]
    Indices:
    a = i*m + j
    b = j*p + k
    c = k*n + i
    """
    nm, mp, pn = n*m, m*p, p*n
    if not U or not V or not W:
        return False
    
    rank = len(U[0])
    
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
                a = i*m + j
                b = j*p + k
                c = k*n + i
                valid.add((a, b, c))
    
    # Check all entries
    # To speed up, we can use numpy if available, but loops are fine for small sizes.
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = 0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
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
    
    # Strassen terms for M_{2,2,2}
    # u, v, w are vectors of length 4 (flattened 2x2 matrices: [00, 01, 10, 11])
    # The tensor identity is sum_r u_r v_r w_r = C
    # where C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Strassen uses 7 multiplications.
    terms = [
        # M1 = (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A10+A11)B00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3 = A00(B10-B11)
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4 = A11(B01-B00)
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5 = (A00+A01)B11
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (A10-A00)(B00+B01)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7 = (A01-A11)(B10+B11)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    # 4 blocks of 2x2 in the 2x8 matrix B
    # B is 2x8. Split into 4 blocks of 2x2: columns (0,1), (2,3), (4,5), (6,7)
    # C is 2x8. Split into 4 blocks of 2x2: columns (0,1), (2,3), (4,5), (6,7)
    
    for block_idx in range(4):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = terms[term_idx]
            
            # Fill U
            # U index is i (0..3) corresponding to A_{i//2, i%2}
            # u is length 4, indexed by local (i, j)
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
    """
    Entry point: return the best decomposition found.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_decomp = None
    best_rank = 32

    # Try rank 28 (Strassen)
    try:
        U, V, W = strassen_decomposition_228()
        if verify(U, V, W):
            best_rank = 28
            best_decomp = {"U": U, "V": V, "W": W}
    except Exception as e:
        print(f"Strassen decomposition failed: {e}")

    # Fallback to schoolbook if rank 28 failed or wasn't tried
    if best_decomp is None:
        try:
            U, V, W = schoolbook()
            if verify(U, V, W):
                best_decomp = {"U": U, "V": V, "W": W}
            else:
                # Should not happen, but return something valid
                best_decomp = {"U": U, "V": V, "W": W}
        except Exception as e:
            print(f"Schoolbook decomposition failed: {e}")
            # Last resort: empty or minimal valid structure? 
            # The task requires exact decomposition. If we fail, we should return something 
            # that might be scored 0 or try to fix. But schoolbook is trivially correct.
            # We assume schoolbook works.

    if time.time() > deadline:
        return best_decomp

    return best_decomp
