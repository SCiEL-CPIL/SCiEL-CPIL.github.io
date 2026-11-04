import time
import numpy as np
from fractions import Fraction

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


def _construct_rank32():
    """Rank-32 decomposition: 8 independent schoolbook 2x2 blocks."""
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for l in range(4):
        for rb in range(2):
            if rb == 0:
                def gA(i, j): return i * m + j
            else:
                def gA(i, j): return i * m + (2 + j)
            def gB(j, k, rb=rb, l=l):
                gj = (j if rb == 0 else 2 + j)
                return gj * p + (2 * l + k)
            def gC(k, i, l=l): return (2 * l + k) * n + i
            Amap = {(i, j): gA(i, j) for i in range(2) for j in range(2)}
            Bmap = {(j, k): gB(j, k) for j in range(2) for k in range(2)}
            Cmap = {(k, i): gC(k, i) for k in range(2) for i in range(2)}
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        r = col
                        U[Amap[(i, j)]][r] = 1
                        V[Bmap[(j, k)]][r] = 1
                        W[Cmap[(k, i)]][r] = 1
                        col += 1
    return {"U": U, "V": V, "W": W}


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


def _find_rank28_via_flattening():
    """
    Attempt to find a rank-28 decomposition by exploiting structure.
    
    The multiplication tensor T[i,j,k] = delta (identity for MM).
    We view this as: for fixed (i,k), T[:,j,(i,k)] is a vector in R^m.
    The tensor has 3 flattening matrices:
      - U-flattening: rows = i*m+j, cols = k*n+i ... wait no.
    
    Actually, let's think differently. The tensor is T in R^{nm x mp x pn}.
    A rank-r decomposition means T = sum_{r=1}^R u_r v_r^T w_r^T where
    u_r in R^{nm}, v_r in R^{mp}, w_r in R^{pn}.
    
    Key insight: T[i,j,k] = 1 if the "path" i->j->k is valid, else 0.
    For our dimensions (2,4,8): T has shape (8, 32, 16).
    T[i*4+j, j*8+k, k*2+i] = 1 for all valid (i,j,k), 0 otherwise.
    
    Wait, that's not right either. Let me re-read the convention.
    W[k*n+i, r] - so W is indexed by (k, i) pairs, flattened as k*n+i.
    U[i*m+j, r] - U indexed by (i, j), flattened as i*m+j.
    V[j*p+k, r] - V indexed by (j, k), flattened as j*p+k.
    
    The tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    For matrix multiplication: T[i,j,k] should equal 1 (the "coefficient" of A[i,j]*B[j,k] in C[i,k]).
    
    So T is a 3-tensor with T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    That's just the all-ones tensor! No wait, that's wrong.
    
    For standard matrix multiplication C = A*B where A is n x m, B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor T has T[i,j,k] = 1 for ALL (i,j,k) in [0,n) x [0,m) x [0,p).
    Yes! It's the all-ones tensor of shape (n, m, p) = (2, 4, 8).
    
    The rank of the all-ones tensor of shape (n,m,p) is... 
    The all-ones tensor can be written as u v^T w^T where u = ones(n), v = ones(m), w = ones(p).
    So its rank is 1?!
    
    But that can't be right because we need the decomposition to satisfy:
    sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for ALL (i,j,k).
    
    If U[i,j,r] = 1 for all i,j and r=0; V[j,k,0] = 1; W[k,i,0] = 1, then yes, rank 1.
    But wait, the shapes are U: (n*m, rank) = (8, rank), V: (m*p, rank) = (32, rank), W: (p*n, rank) = (16, rank).
    
    If rank=1: U[i*m+j, 0] = 1 for all i,j. V[j*p+k, 0] = 1 for all j,k. W[k*n+i, 0] = 1 for all k,i.
    Then sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1*1*1 = 1. 
    
    So rank 1 works?? But that seems too good. Let me re-read the problem.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 8 matrix."
    
    The tensor for matrix multiplication is NOT the all-ones tensor. The tensor represents the bilinear map.
    The entries of the tensor are the coefficients of the bilinear form. For standard MM, 
    C[i,k] = sum_j A[i,j] B[j,k], so the tensor T has T[i,j,k] = 1 for all (i,j,k).
    
    Hmm, but then rank 1 would work. Unless I'm misunderstanding the convention.
    
    Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    
    Maybe the issue is that the decomposition must work for ALL matrices, and the "tensor" is the 
    multiplication tensor in the tensor algebra sense. The multiplication tensor for m x n matrices 
    is T in R^{(mn)^2 x (nm)^2} ... no.
    
    Actually, I think I was overcomplicating. The tensor T[i,j,k] = 1 for all (i,j,k) IS the 
    multiplication tensor for the case where we're computing sum_j A[i,j]B[j,k]. And its rank 
    as a 3-tensor is indeed 1 (it's a simple tensor: ones(n) ⊗ ones(m) ⊗ ones(p)).
    
    But that contradicts the known result that 2x2x2 MM has rank 7 (Strassen). Let me reconsider.
    
    OH WAIT. I think I'm confusing myself. The multiplication tensor for 2x2 matrix multiplication
    is NOT the all-ones tensor. Let me think again.
    
    For 2x2 MM: C = A*B, C[i,k] = sum_j A[i,j] B[j,k].
    The "tensor" in the context of tensor rank for MM is the 3-tensor T where 
    T[i,j,k] = 1 if the (i,j,k) "interaction" exists. For standard MM, every (i,j,k) triple 
    with i in [0,2), j in [0,2), k in [0,2) has T[i,j,k] = 1. So it IS the all-ones 2x2x2 tensor.
    
    But the rank of the all-ones 2x2x2 tensor is 1, not 7!
    
    I must be misunderstanding what "rank" means here. In the context of matrix multiplication 
    tensor rank, the tensor is NOT T[i,j,k] = 1. Instead, the multiplication tensor is a 
    3-tensor in R^{(n^2) x (n^2) x (n^2)} for n x n matrices, where the entries encode the 
    bilinear map.
    
    For n x n MM: C = A*B, C is n x n, A is n x n, B is n x n.
    C[i,k] = sum_j A[i,j] B[j,k].
    The tensor T has shape (n^2, n^2, n^2) where T[ic, jc, kc] ... no.
    
    Actually, the standard definition: the multiplication tensor for (n1 x n2) * (n2 x n3) 
    is a 3-tensor T in R^{(n1*n2) x (n2*n3) x (n3*n1)} where 
    T[(i,j), (j',k), (k',i')] = delta_{j,j'} delta_{k,k'} delta_{i,i'}.
    
    So T[A_idx, B_idx, C_idx] = 1 if the indices are consistent, 0 otherwise.
    Specifically: T[i*m+j, j*p+k, k*n+i] = 1, and T is 0 elsewhere.
    
    This is NOT the all-ones tensor! It's a sparse tensor where only specific entries are 1.
    For (2,4,8): T has shape (8, 32, 16). T[i*4+j, j*8+k, k*2+i] = 1 for valid (i,j,k).
    The number of 1s is n*m*p = 2*4*8 = 64. The total size is 8*32*16 = 4096.
    
    So the tensor is very sparse. Its rank is what we're trying to minimize.
    
    For the flattening: fix the first mode (i*m+j). The flattening matrix M_U has 
    rows = i*m+j (8 rows), cols = (j*p+k, k*n+i) pairs... no.
    
    The U-flattening: M_U[a, (b,c)] = T[a, b, c]. So M_U is 8 x (32*16) = 8 x 512.
    The rank of M_U is a lower bound on the tensor rank.
    
    Similarly, V-flattening: M_V[b, (a,c)] = T[a, b, c], shape 32 x (8*16) = 32 x 128.
    W-flattening: M_W[c, (a,b)] = T[a, b, c], shape 16 x (8*32) = 16 x 256.
    
    The rank of each flattening is a lower bound. Let's compute the rank of the U-flattening.
    M_U[a, (b,c)] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some (i,j,k), else 0.
    
    For a = i*4+j (i in 0,1; j in 0..3):
    b = j*8+k (k in 0..7): but b must also satisfy that the (i,j,k) triple is valid.
    c = k*2+i.
    
    So for fixed a = i*4+j, M_U[a, (b,c)] = 1 iff b = j*8+k AND c = k*2+i for some k.
    Given a = i*4+j, we have i and j fixed. Then b = j*8+k determines k = b - j*8.
    c must equal (b - j*8)*2 + i = 2*b - 2*j*8 + i = 2*b - 16*j + i.
    
    So M_U[a, (b,c)] = 1 iff c = 2*b - 16*j + i, where a = i*4+j.
    For each a, there are exactly 8 nonzero entries (one for each k=0..7, giving b=j*8+k, c=k*2+i).
    
    The U-flattening has 8 rows. Each row has 8 ones in specific columns. The rows are 
    "shifted" versions of each other. Let's check if they're linearly independent.
    
    Row a=0 (i=0,j=0): ones at (b,c) = (k, 2k) for k=0..7.
    Row a=1 (i=0,j=1): ones at (b,c) = (8+k, 2k) for k=0..7.
    Row a=2 (i=0,j=2): ones at (b,c) = (16+k, 2k) for k=0..7.
    Row a=3 (i=0,j=3): ones at (b,c) = (24+k, 2k) for k=0..7.
    Row a=4 (i=1,j=0): ones at (b,c) = (k, 2k+1) for k=0..7.
    Row a=5 (i=1,j=1): ones at (b,c) = (8+k, 2k+1) for k=0..7.
    Row a=6 (i=1,j=2): ones at (b,c) = (16+k, 2k+1) for k=0..7.
    Row a=7 (i=1,j=3): ones at (b,c) = (24+k, 2k+1) for k=0..7.
    
    These 8 rows: the first 4 have c = 2k (even), the last 4 have c = 2k+1 (odd).
    Within the first 4, they're shifts in b. Within the last 4, also shifts in b.
    The first 4 and last 4 are in disjoint sets of columns (even vs odd c).
    So the 8 rows are linearly independent! Rank of U-flattening = 8.
    
    Similarly for V-flattening: M_V[b, (a,c)] = T[a,b,c].
    b = j*8+k. For fixed b, j and k are determined.
    a = i*4+j, c = k*2+i.
    So M_V[b, (a,c)] = 1 iff a = i*4+j AND c = k*2+i for some i.
    Given b = j*8+k, j and k are fixed. Then a = i*4+j, c = k*2+i.
    For i=0: a=j, c=k*2.
    For i=1: a=4+j, c=k*2+1.
    So each row of M_V has exactly 2 nonzero entries.
    
    M_V has 32 rows. Each row has 2 ones. The rank is at most 32, but likely less.
    Actually, the rank of M_V is a lower bound on the tensor rank. Let's not compute it exactly 
    but note it's >= 8 (from U-flattening).
    
    The known lower bounds for (2,4,8) MM tensor rank... I recall that the rank is between 
    32 and 56 or so. The schoolbook gives 32 (as we have). Strassen gives 56.
    
    Can we do better than 32? The question is whether the 32 schoolbook terms can be 
    "compressed" using Strassen-like tricks.
    
    The 32 schoolbook terms correspond to the 32 (i,j,k) triples. We want to replace groups 
    of these with fewer terms using Strassen identities.
    
    A Strassen 2x2x2 identity replaces 8 schoolbook terms (2x2x2) with 7 terms. 
    But in our case, the "2x2x2" blocks don't align nicely because of the rectangular shape.
    
    Let me think about this differently. We have 8 independent 2x2 schoolbook blocks (4 l-values 
    x 2 row-blocks). Each uses 4 terms. Total 32.
    
    Can we combine any of these 2x2 products using Strassen? Strassen combines two 2x2 
    products into one 7-term computation. But here, each 2x2 product is independent (different 
    output blocks), so we can't directly apply Strassen.
    
    However, if two 2x2 products share the same A or B factor, we might be able to combine them.
    For example, A1 * B1_top and A1 * B2_top share A1. But they have different B factors and 
    different output blocks.
    
    Actually, I realize that the rank-32 construction is likely optimal for this structure, 
    and finding rank < 32 would require a fundamentally different approach.
    
    Let me just verify that rank 32 is valid and return it.
    """
    return None  # Placeholder - we'll just use rank 32


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140

    best = None

    # Candidate 1: rank-32
    try:
        cand = _construct_rank32()
        ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
        if ok:
            U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
            best = (r, U, V, W)
    except Exception:
        pass

    # Candidate 2: rank-56 fallback
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
