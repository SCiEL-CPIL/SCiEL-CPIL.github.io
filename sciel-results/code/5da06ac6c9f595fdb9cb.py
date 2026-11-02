"""Search for a low-rank exact tensor decomposition of (2,2,8) matrix multiplication."""
import time
import itertools
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

def try_rank_24():
    """
    Attempt to find a rank-24 decomposition by exploiting the block structure.
    Since P=8 is a multiple of 4, we can group columns into 2 groups of 4.
    Within each group of 4 columns, we have a 2x2x4 tensor.
    The 2x2x4 tensor has rank 14 (known result: R(M_{2,2,4}) = 14).
    So 2 * 14 = 28, which doesn't help directly.
    
    However, we can try a different approach: use the fact that the 2x2x8 tensor
    can be viewed as two copies of 2x2x4, and see if we can share some rank-1 terms
    between the two copies.
    
    This is complex, so we'll try a simpler heuristic search for rank < 28.
    """
    deadline = time.time() + 60  # spend at most 60s on this
    
    # Start from the rank-28 Strassen decomposition
    U, V, W = strassen_decomposition_228()
    
    if not verify(U, V, W):
        return None
    
    # Try to reduce rank by removing columns that are linearly dependent
    # or by finding linear combinations
    rank = 28
    
    # Build the tensor T as a vector for analysis
    nm, mp, pn = NM, MP, PN
    valid = set()
    for i in range(N):
        for j in range(M):
            for k in range(P):
                valid.add((i*M+j, j*P+k, k*N+i))
    
    # Get all tensor entries
    tensor_entries = {}
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) in valid:
                    tensor_entries[(a, b, c)] = 1
    
    # Try removing pairs of columns and checking if the remaining can still
    # span the tensor by adjusting weights
    # This is a heuristic: check if any column is a linear combination of others
    
    # For now, return None to indicate no improvement found
    return None

def construct():
    """Entry point: return the best decomposition found."""
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # First try the known rank-28 Strassen decomposition
    try:
        U, V, W = strassen_decomposition_228()
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Try to find a rank-24 decomposition
    try:
        result = try_rank_24()
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
