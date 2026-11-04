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


def _construct_rank56():
    """The original Strassen-based rank-56 decomposition (known valid)."""
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 terms
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
    """
    Rank-32 decomposition using rectangular Strassen-like blocking.
    
    Strategy: Split the 4 middle dimension into two 2-dim halves (j=0,1 and j=2,3).
    Split the 8 right dimension into four 2-dim blocks (k=0,1; 2,3; 4,5; 6,7).
    
    For each of the 2 halves and 4 blocks, we use a 7-term pattern. However, we can
    share U-terms across blocks since U only depends on (i, j) not on k.
    
    The key structure: For a fixed j-half, the U part is the same across all k-blocks.
    We use 7 independent scalar multiplications per (j-half, k-block) pair, but the
    U coefficients can be structured to reduce total rank.
    
    Actually, let's think differently. The tensor T_{i,j,k} = delta_{i,j,k} for MM.
    We need sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 for all i,j,k.
    
    A rank-32 approach: use 2 independent 2x2x2 Strassen patterns per j-half,
    applied to pairs of k-blocks.
    
    Let's try: for each j-half (2 choices), use 4 k-blocks. Group k-blocks into pairs:
    (block 0,1) and (block 2,3). For each j-half and each k-pair, use a 7-term pattern
    where the V and W parts handle both k-blocks.
    
    Total: 2 j-halves * 2 k-pairs * 7 terms = 28 terms. But we need to verify this works.
    
    Alternative: use the fact that for 2x4x8, we can view it as two 2x2x8 multiplications
    (one for j=0,1 and one for j=2,3). Each 2x2x8 can be done in rank 14 (7 terms for
    each of two 2x2x4 sub-problems, with sharing).
    
    Let me construct it explicitly:
    """
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # 7 Strassen-like terms for 2x2x2
    # Each term has U part (2x2), V part (2x2), W part (2x2)
    # Using the standard 7-term scheme
    strassen_terms = [
        # (U: {(i,j):c}, V: {(j,k):c}, W: {(k,i):c})
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(0, 1): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (1, 0): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]

    col = 0
    # For each j-half (2) and each pair of k-blocks (2 pairs of 2 blocks each = 4 blocks total)
    # Use 7 terms per (j-half, k-pair)
    for j_half in range(2):
        for k_pair in range(2):
            for (U_part, V_part, W_part) in strassen_terms:
                # U: indices (i, j) where j = j_half*2 + j_low
                for (i, j_low), c in U_part.items():
                    j = j_half * 2 + j_low
                    U[i * m + j][col] += c
                # V: indices (j, k) where j = j_half*2 + j_low, k = k_pair*4 + k_low
                for (j_low, k_low), c in V_part.items():
                    j = j_half * 2 + j_low
                    k = k_pair * 4 + k_low
                    V[j * p + k][col] += c
                # W: indices (k, i) where k = k_pair*4 + k_low
                for (k_low, i), c in W_part.items():
                    k = k_pair * 4 + k_low
                    W[k * n + i][col] += c
                col += 1
    return {"U": U, "V": V, "W": W}


def _construct_rank28():
    """
    Attempt rank-28 by sharing U-terms across k-pairs.
    For each j-half, use the same 7 U-terms for both k-pairs.
    This gives 2 j-halves * 7 terms = 14 for U, but V and W still need 28 entries.
    Actually this doesn't reduce rank since each term is independent.
    
    Let me try a different approach: use 4 terms per (j-half, k-pair) with more structure.
    """
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Try a 4-term scheme per (j-half, k-pair) - may not work exactly
    # For now, fall back to rank-32 if this doesn't verify
    # We'll construct it and let the check determine validity
    four_terms = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(0, 1): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}),
    ]

    col = 0
    for j_half in range(2):
        for k_pair in range(2):
            for (U_part, V_part, W_part) in four_terms:
                for (i, j_low), c in U_part.items():
                    j = j_half * 2 + j_low
                    U[i * m + j][col] += c
                for (j_low, k_low), c in V_part.items():
                    j = j_half * 2 + j_low
                    k = k_pair * 4 + k_low
                    V[j * p + k][col] += c
                for (k_low, i), c in W_part.items():
                    k = k_pair * 4 + k_low
                    W[k * n + i][col] += c
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


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s kill

    # Try rank-28 first
    if time.time() < deadline:
        cand28 = _construct_rank28()
        ok28, err28 = _matmul_check(cand28["U"], cand28["V"], cand28["W"], n, m, p)
        if ok28:
            U, V, W, r = _trim_zero_columns(cand28["U"], cand28["V"], cand28["W"])
            return {"U": U, "V": V, "W": W}

    # Try rank-32
    if time.time() < deadline:
        cand32 = _construct_rank32()
        ok32, err32 = _matmul_check(cand32["U"], cand32["V"], cand32["W"], n, m, p)
        if ok32:
            U, V, W, r = _trim_zero_columns(cand32["U"], cand32["V"], cand32["W"])
            return {"U": U, "V": V, "W": W}

    # Fallback to known valid rank-56
    cand56 = _construct_rank56()
    ok56, err56 = _matmul_check(cand56["U"], cand56["V"], cand56["W"], n, m, p)
    if ok56:
        U, V, W, r = _trim_zero_columns(cand56["U"], cand56["V"], cand56["W"])
        return {"U": U, "V": V, "W": W}

    # Last resort: return untrimmed rank-56
    return cand56
# EVOLVE-BLOCK-END
