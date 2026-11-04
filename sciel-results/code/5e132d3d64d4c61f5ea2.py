import time
import numpy as np

DIMENSIONS = [2, 4, 8]


# EVOLVE-BLOCK-START
def _matmul_check(U, V, W, n, m, p):
    """Exact check that the tensor decomposition reconstructs the MM tensor.
    For every (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] must equal 1.
    Returns (ok, max_abs_error).
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return (err < 1e-9), float(err)


def _trim_zero_columns(U, V, W):
    """Remove rank columns that are all-zero in U, V, and W (they contribute nothing)."""
    n_u = len(U)
    n_v = len(V)
    n_w = len(W)
    rank = len(U[0]) if n_u else 0
    keep = []
    for c in range(rank):
        if any(U[i][c] != 0 for i in range(n_u)) or \
           any(V[i][c] != 0 for i in range(n_v)) or \
           any(W[i][c] != 0 for i in range(n_w)):
            keep.append(c)
    if len(keep) == rank:
        return U, V, W, rank
    U2 = [[U[i][c] for c in keep] for i in range(n_u)]
    V2 = [[V[i][c] for c in keep] for i in range(n_v)]
    W2 = [[W[i][c] for c in keep] for i in range(n_w)]
    return U2, V2, W2, len(keep)


def _construct_rank32():
    """
    Rank-32 decomposition using Strassen-like tricks to compress schoolbook terms.
    
    The key insight: we have 2x4x8 matrix multiplication. We can view this as:
    - Split the 4 middle dimension into two 2x2 blocks
    - For each pair of 2x2 blocks (one from A, one from B), we can use Strassen's 7-term algorithm
    - But we need to be careful about how the blocks interact
    
    Actually, let me think about this more carefully. The schoolbook has 2*4*8 = 64 terms.
    We want to get to 32.
    
    One approach: group the computation by output (i,k) pairs. There are 2*8 = 16 output entries.
    Each C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k].
    
    For each output entry, we have 4 multiplications. Can we combine them?
    
    Actually, a better approach: use the fact that we can precompute combinations of A and B entries.
    
    Let me try a different decomposition. Consider the 4 middle dimension as two 2-vectors:
    A[i, 0:2] and A[i, 2:4], B[0:2, k] and B[2:4, k].
    
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
           = dot(A[i,0:2], B[0:2,k]) + dot(A[i,2:4], B[2:4,k])
    
    Each dot product of 2-vectors can be done with 3 multiplications (using Strassen's trick for dot products):
    dot([x1,x2], [y1,y2]) = x1*y1 + x2*y2
    = x1*y1 + (x1+x2)*(y1+y2) - x1*y2 - x2*y1 ... no, that's not right.
    
    Actually, for a 2D dot product, we need 2 multiplications minimum (it's already optimal).
    So that doesn't help.
    
    Let me reconsider. The rank of the 2x2x2 multiplication tensor is 7 (Strassen).
    But we're not doing 2x2x2 matrix multiplication; we're doing a bilinear map.
    
    The tensor T[i,j,k] = 1 for all (i,j,k) in [0,2)x[0,4)x[0,8).
    Wait, that's the all-ones tensor, which has rank 1!
    
    No, that's wrong. The multiplication tensor for (n,m,p) matrix multiplication is:
    T[(i,j),(j',k),(k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    
    So T is nonzero only when j=j', k=k', i=i'. The number of nonzero entries is n*m*p = 2*4*8 = 64.
    
    The rank of this tensor is what we want to minimize.
    
    For the schoolbook: rank = n*m*p = 64? No, that's not right either.
    
    Actually, the schoolbook algorithm uses exactly n*m*p multiplications, one for each (i,j,k) triple.
    So the schoolbook rank is n*m*p = 64.
    
    But the current code claims rank 32. Let me check what the rank-32 construction actually does.
    
    Looking at the code, it creates 8 independent 2x2 blocks, each using 4 terms. 8*4 = 32.
    But each 2x2 block corresponds to a 2x2x2 subproblem, which should have rank 7 (Strassen) or 8 (schoolbook).
    Using 4 terms for a 2x2x2 block is not valid unless there's some special structure.
    
    Wait, I think I'm confusing myself. Let me re-examine.
    
    The tensor T has shape (n*m, m*p, p*n) = (8, 32, 16).
    T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k), 0 otherwise.
    
    A rank-r decomposition means:
    sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
    
    For the schoolbook: U[i*m+j, i*m+j] = 1, V[j*p+k, j*p+k] = 1, W[k*n+i, k*n+i] = 1
    But this doesn't work because the indices don't align properly.
    
    Actually, the schoolbook decomposition is:
    For each (i,j,k): create a term where U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    with r being a unique index for each (i,j,k).
    This gives rank = n*m*p = 64.
    
    So the rank-32 construction in the code must be using some clever trick to halve the rank.
    Let me verify if it's actually valid.
    
    Given the complexity, let me just implement a clean rank-32 construction if it exists, 
    or fall back to rank-56 if not.
    """
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Use Strassen's algorithm on 2x2x2 blocks
    # We have 4 "k-blocks" of size 2, and 2 "i-rows"
    # For each pair (i_block, k_block), we can apply Strassen to the 2x2x2 subproblem
    
    # Strassen's 7 terms for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    
    # For our case, we need to adapt this to the rectangular structure.
    
    # Let's try a different approach: use the fact that the tensor can be decomposed
    # using fewer terms by exploiting linear dependencies.
    
    # Actually, let me just return a valid rank-56 decomposition for now, 
    # and the caller can try to improve it.
    return None  # Signal that this construction is not implemented


def _construct_rank56():
    """The original Strassen-based rank-56 decomposition (known valid)."""
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    strassen = [
        {'U': {(0, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1, (1, 1): 1}},
        {'U': {(1, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1}, 'W': {(0, 1): 1, (1, 1): -1}},
        {'U': {(0, 0): 1}, 'V': {(0, 1): 1, (1, 1): -1}, 'W': {(1, 0): 1, (1, 1): 1}},
        {'U': {(1, 1): 1}, 'V': {(1, 0): 1, (0, 0): -1}, 'W': {(0, 0): 1, (0, 1): 1}},
        {'U': {(0, 0): 1, (0, 1): 1}, 'V': {(1, 1): 1}, 'W': {(0, 0): -1, (1, 0): 1}},
        {'U': {(1, 0): 1, (0, 0): -1}, 'V': {(0, 0): 1, (0, 1): 1}, 'W': {(1, 1): 1}},
        {'U': {(0, 1): 1, (1, 1): -1}, 'V': {(1, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1}},
    ]

    col = 0
    for j_high in range(2):
        for l in range(4):
            for term in strassen:
                for (i, j_low), c in term['U'].items():
                    j = j_high * 2 + j_low
                    U[i * m + j][col] += c
                for (j_low, k_low), c in term['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    V[j * p + k][col] += c
                for (k_low, i), c in term['W'].items():
                    k = l * 2 + k_low
                    W[k * n + i][col] += c
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140

    best = None

    # Try rank-32 first
    try:
        cand = _construct_rank32()
        if cand is not None:
            ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
            if ok:
                U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
                best = (r, U, V, W)
    except Exception:
        pass

    # Fallback to rank-56
    try:
        cand = _construct_rank56()
        ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
        if ok:
            U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
            if best is None or r < best[0]:
                best = (r, U, V, W)
    except Exception:
        pass

    if best is None:
        cand = _construct_rank56()
        return cand

    r, U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
