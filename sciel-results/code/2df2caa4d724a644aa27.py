"""Matrix multiplication 4x4 * 4x9 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 138.0  # safety margin below the 150s kill


def _build_tensor(n, m, p):
    """T[io, u, v] = 1 iff output (i,k) uses a[i,j]*b[j,k]; io=k*n+i, u=i*m+j, v=j*p+k."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(k * n + i, i * m + j, j * p + k)] = 1
    return T


def _exact_check(U, V, W, T, n, m, p):
    """Verify sum_c U[u,c]*V[v,c]*W[io,c] == T[io,u,v] for all (io,u,v)."""
    rank = len(U[0])
    for io in range(p * n):
        for u in range(n * m):
            for v in range(m * p):
                s = 0
                for c in range(rank):
                    s += U[u][c] * V[v][c] * W[io][c]
                if s != T.get((io, u, v), 0):
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _rank36_decomposition(n, m, p):
    """
    Attempt a rank-36 decomposition using structured factorization.
    
    The idea: for matrix multiplication C = A * B where A is n×m and B is m×p,
    we can use the fact that the tensor has a special structure.
    
    For rank 36 with n=4, m=4, p=9:
    We need 36 factors. Each factor corresponds to a "path" through the indices.
    
    Strategy: Use the structure that for each (j, k) pair in B and each i in A,
    we can potentially share computation.
    
    Actually, let me think about this differently. The lower bound is 36 = m*p = 4*9.
    This suggests we might be able to achieve rank 36 by having one factor per (j,k) pair
    with appropriate weighting across i indices.
    
    For each (j, k) in range(m) x range(p), we create a factor c.
    U[i*m+j][c] = 1 for all i (since A[i,j] is used for all i)
    V[j*p+k][c] = 1
    W[k*n+i][c] = 1 for all i
    
    Wait, that gives us m*p = 36 factors, but we need to check if this works.
    For a given output C[i,k] = sum_j A[i,j] * B[j,k]:
    sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
    
    With the above scheme, for fixed (i,k), the sum becomes:
    sum_j U[i*m+j][j*p+k] * V[j*p+k][j*p+k] * W[k*n+i][j*p+k]
    = sum_j 1 * 1 * 1 = m = 4
    
    But we need C[i,k] = sum_j A[i,j]*B[j,k], so this gives 4, not the product.
    This doesn't work directly.
    
    Let me try a different approach. We need the tensor to decompose as:
    T[io,u,v] = sum_c U[u,c] * V[v,c] * W[io,c]
    
    where T[k*n+i, i*m+j, j*p+k] = 1 and 0 otherwise.
    
    For rank 36, we have 36 factors. Let's index them by (j, k) where j in [0,m), k in [0,p).
    
    For factor c=(j,k):
    - It should contribute to all (i, j, k) triples where the indices match
    - U[i*m+j][c] should be non-zero only when the row index matches
    - V[j'*p+k'][c] should be non-zero only when the column index matches  
    - W[k'*n+i'][c] should be non-zero only when the output index matches
    
    Actually, the correct structure for a rank-m*p decomposition would be:
    For each (j, k), create factor c:
    - U[u][c] = 1 if u = i*m+j for some i, else 0 → but this depends on i
    - This doesn't work because U can't depend on i independently
    
    Let me reconsider. The standard approach for achieving the lower bound uses
    the fact that matrix multiplication can be viewed as a sum of outer products.
    
    Actually, for this specific case, let me try to use the identity:
    A*B = sum_j (A[:,j] * B[j,:])
    
    This gives m=4 terms, but each term is a rank-1 matrix (outer product of vectors).
    The tensor rank of each outer product A[:,j] * B[j,:] is p*n = 36 (since it's a 
    4x9 matrix). So total rank would be 4*36 = 144, which is the schoolbook.
    
    To get rank 36, we need a more clever decomposition. Let me try using the 
    structure of the problem more carefully.
    
    For n=4, m=4, p=9, the lower bound is 36. Let me try to construct it.
    
    One approach: use the fact that we can write the multiplication as a sum of
    rank-1 tensors. The key is to find 36 tensors whose sum equals the multiplication tensor.
    
    Let me try: for each (j, k) pair, define a tensor T_{j,k} such that:
    T_{j,k}[io, u, v] = 1 if u = i*m+j for some i, v = j*p+k, and io = k*n+i
    for the appropriate i, else 0.
    
    But this is exactly the schoolbook with one factor per (i,j,k), giving 144 factors.
    
    To reduce to 36, we need to combine the i dimension. For fixed (j,k), we have
    n=4 values of i. Can we represent the sum over i as a single rank-1 tensor?
    
    For fixed (j,k), the contribution to the tensor is:
    sum_i delta(u = i*m+j) * delta(v = j*p+k) * delta(io = k*n+i)
    
    This is a sum of n rank-1 tensors. Can we write it as a single rank-1 tensor?
    Only if the sum itself is rank 1, which it's not in general.
    
    So rank 36 might not be achievable with simple integer entries. Let me try a 
    different strategy: use the structure of the problem to find a valid decomposition.
    
    Actually, I recall that for matrix multiplication, the rank is known to be 
    at most n*m*p and at least max(n*m, m*p, p*n). For some dimensions, the 
    minimum rank is known to be strictly between these bounds.
    
    For 4x4 * 4x9, let me try to search for a rank-36 decomposition using 
    a structured approach. If it fails, I'll fall back to schoolbook.
    """
    rank = m * p  # 36
    U = [[0] * rank for _ in range(n * m)]  # 16 x 36
    V = [[0] * rank for _ in range(m * p)]  # 36 x 36
    W = [[0] * rank for _ in range(p * n)]  # 36 x 36
    
    # For each (j, k) pair, assign factor index c = j * p + k
    # U[i*m+j][c] = 1 for all i (since A[i,j] contributes to C[i,k])
    # V[j*p+k][c] = 1
    # W[k*n+i][c] = 1 for all i
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            # V: only the (j,k) entry is 1
            V[j * p + k][c] = 1
            # U: all rows i for column j
            for i in range(n):
                U[i * m + j][c] = 1
            # W: all columns i for row k
            for i in range(n):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # First try the rank-36 structured decomposition
    candidate = _rank36_decomposition(n, m, p)
    if _exact_check(candidate["U"], candidate["V"], candidate["W"], T, n, m, p):
        return candidate
    
    # Fall back to schoolbook
    best = _schoolbook(n, m, p)
    if _exact_check(best["U"], best["V"], best["W"], T, n, m, p):
        return best
    
    return best
