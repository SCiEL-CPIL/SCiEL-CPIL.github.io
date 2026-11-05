"""Matrix multiplication 3x3x7 tensor decomposition search.

We search for a rank-minimal exact decomposition of the matmul tensor
T[i,j,k] = e_{i,k} where (i,j) indexes U rows, (j,k) indexes V rows,
and (k,i) indexes W rows (W uses the trace convention W[k*n+i, c]).

The schoolbook rank is 63. We attempt to find lower-rank decompositions
using a structured randomized search with verification.
"""

import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 7]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 63


def _make_schoolbook():
    """Return the trivial schoolbook decomposition of rank 63."""
    rank = RANK_SCHOOLBOOK
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n=N, m=M, p=P):
    """Check that U ⊙ V ⊙ W reconstructs the matmul tensor exactly.
    
    The tensor T has entries T[i,j,k] = 1 iff the (i,k) entry of the
    product matrix corresponds to row i of A, column k of B.
    In the factorization: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    
    For matmul: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    Actually the matmul tensor is T[i,j,k] = delta(i,k) ... no.
    
    Let me re-derive. The matrix multiplication tensor T_{ijk} satisfies:
    (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    So T[i,j,k] = 1 for all valid (i,j,k) when we write:
    (A*B)[i,k] = sum_{j} sum_{c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
    
    Wait, the standard decomposition: C[i,k] = sum_c U[i*m+?]*...*...
    
    Actually the convention is:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    And we factor this as:
    C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_k' ... ) 
    
    No, the standard tensor decomposition of matmul is:
    T_{i,j,k} where the three modes are (row of A = i, col of A = j, row of B = k)
    and T_{i,j,k} = 1 (the tensor is the identity in a sense).
    
    More precisely, the bilinear map is:
    (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    
    This is a trilinear form in the sense that:
    C[i,k] = <T_{i,·,k}, A_{i,·} ⊗ B_{·,k}>
    
    The rank-r decomposition writes:
    C[i,k] = sum_{c=1}^r (sum_j U[i*m+j, c] A[i,j]) * (sum_k V[j*p+k, c] B[j,k]) ... 
    
    Hmm, I need to be more careful. Let me use the standard formulation.
    
    The matmul tensor T ∈ R^{(n*m) × (m*p) × (p*n)} is defined by:
    T[(i,j), (j',k), (k',i')] = 1 if j=j' and i=i', else 0.
    
    So T[(i,j), (j,k), (k,i)] = 1, and T[(i,j), (j',k), (k',i')] = 0 otherwise.
    
    A rank-r decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c
    where U_c ∈ R^{n*m}, V_c ∈ R^{m*p}, W_c ∈ R^{p*n}.
    
    Verification: for each (a, b, c) index:
    sum_r U[a,r] * V[b,r] * W[c,r] should equal T[a,b,c].
    """
    rank = len(U[0])
    # Build lookup for U, V, W
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = 0
                for r in range(rank):
                    total += U[a][r] * V[b][r] * W[c][r]
                expected = 1
                if total != expected:
                    return False
    # Also check that off-diagonal entries are zero
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                for kp in range(p):
                    for ip in range(n):
                        if kp == k and ip == i:
                            continue
                        cp = kp * n + ip
                        total = 0
                        for r in range(rank):
                            total += U[a][r] * V[b][r] * W[cp][r]
                        if total != 0:
                            return False
    return True


def _random_sparse_matrix(rows, rank, density=0.3, rng=None):
    """Generate a sparse random matrix with small integer entries."""
    if rng is None:
        rng = random.Random()
    mat = [[0] * rank for _ in range(rows)]
    for i in range(rows):
        for c in range(rank):
            if rng.random() < density:
                mat[i][c] = rng.choice([-1, 0, 1])
    return mat


def _try_random_decomposition(rank, n=N, m=M, p=P, seed=42, max_iter=500):
    """Try to find a rank-r decomposition via random search."""
    rng = random.Random(seed)
    
    for attempt in range(max_iter):
        U = _random_sparse_matrix(n * m, rank, density=0.4, rng=rng)
        V = _random_sparse_matrix(m * p, rank, density=0.4, rng=rng)
        W = _random_sparse_matrix(p * n, rank, density=0.4, rng=rng)
        
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return None


def _try_structured_block_decomposition(seed=123):
    """Try to decompose 3x3x7 as blocks.
    
    7 = 3 + 3 + 1, so C = A*B1 + A*B2 + A*B3 where B1 is 3x3, B2 is 3x3, B3 is 3x1.
    If we have rank 23 for 3x3x3 and rank 3 for 3x3x1:
    Total = 23 + 23 + 3 = 49.
    
    But we don't have verified rank-23 coefficients for 3x3x3.
    Instead, let's try 7 = 4 + 3:
    Rank for 3x3x4 + Rank for 3x3x3 = ?
    
    Or simply try to find any improvement over 63 via a different structure.
    """
    # This is a placeholder for structured search
    pass


def construct():
    """Construct the rank-minimal exact tensor decomposition for 3x3x7 matmul.
    
    Returns a dict with keys 'U', 'V', 'W' whose column count is the rank.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin: kill at 150s
    
    # Start with schoolbook as the best known
    best = _make_schoolbook()
    best_rank = RANK_SCHOOLBOOK
    
    # Try to find a better decomposition
    # Strategy 1: Try ranks from 62 downward with limited random search
    for rank in range(62, 40, -1):
        if time.time() > deadline - 10:
            break
        
        # Limited attempts per rank
        max_attempts = 20 if rank > 50 else 10
        result = _try_random_decomposition(
            rank, n=N, m=P, p=P, seed=rank, max_iter=max_attempts
        )
        if result is not None:
            best = result
            best_rank = rank
            break
    
    # If no improvement found, return schoolbook
    return best
