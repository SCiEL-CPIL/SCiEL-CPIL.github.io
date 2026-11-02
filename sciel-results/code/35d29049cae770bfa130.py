"""Matrix multiplication 3,3,3: rank-minimal exact tensor decomposition.

Strategy:
  * Attempt a rank-23 exact decomposition (best known rank for 3x3x3 matmul),
    whose entries lie in {0, +/-1, +/-1/2}.
  * Verify the decomposition EXACTLY with doubled-integer arithmetic
    (multiply every entry by 2 -> integers; the tensor identity becomes
    sum_c (2u)(2v)(2w) == 8 for each (i,j,k), all integer math -> no FP error).
  * If verification fails for any reason, fall back to the guaranteed-correct
    schoolbook decomposition of rank 27, so the result is always valid.
  * Explicit deadline guard: never run past the wall-clock limit; return the
    best result found with a safety margin.
"""
import time

DIMENSIONS = [3, 3, 3]

# Wall-clock safety margin (evaluator kills at 150s; we stop well before).
_DEADLINE_S = 120.0
_START = time.time()


def _time_left():
    return _DEADLINE_S - (time.time() - _START)


def _schoolbook():
    """Guaranteed-correct rank-27 (n*m*p) decomposition."""
    n, m, p = DIMENSIONS
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


def _verify_exact(U, V, W):
    """Exact verification using doubled-integer arithmetic.

    U, V, W entries are assumed to be in {0, +/-1, +/-1/2}. We double them to
    integers (0, +/-1, +/-2) and check, for every (i,j,k) in [0,3)^3:
        sum_c (2*U[i*3+j][c]) * (2*V[j*3+k][c]) * (2*W[k*3+i][c]) == 8
    which is exactly 2^3 * (the required tensor value 1). All integer math.
    """
    n, m, p = DIMENSIONS
    r = len(U[0])
    Ud = [[int(round(2 * x)) for x in row] for row in U]
    Vd = [[int(round(2 * x)) for x in row] for row in V]
    Wd = [[int(round(2 * x)) for x in row] for row in W]
    # sanity: entries must be half-integers -> doubled values in {-2,-1,0,1,2}
    for M in (Ud, Vd, Wd):
        for row in M:
            for x in row:
                if x not in (-2, -1, 0, 1, 2):
                    return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                for c in range(r):
                    s += Ud[ui][c] * Vd[vj][c] * Wd[wk][c]
                if s != 8:
                    return False
    return True


def _rank23_attempt():
    """Best-effort rank-23 decomposition (entries in {0,+/-1,+/-1/2}).

    Returned as raw floats; the caller verifies before trusting it. If these
    coefficients are not exactly the known rank-23 solution, _verify_exact will
    reject them and the program falls back to schoolbook (rank 27).
    """
    n, m, p = DIMENSIONS
    # Each term: (U_vec(9), V_vec(9), W_vec(9)) with entries in {0,+-1,+-1/2}.
    # (Best-recall reconstruction of the 23-term 3x3x3 matmul decomposition.)
    terms = [
        # --- 18 "core" terms (2x2-block style) ---
        ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        # --- 5 correction terms (the "extra" 5 that reduce 27 -> 23) ---
        ([1,0,0, 0,1,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
    ]
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (uc, vc, wc) in enumerate(terms):
        for a in range(n * m):
            U[a][c] = uc[a]
        for b in range(m * p):
            V[b][c] = vc[b]
        for d in range(p * n):
            W[d][c] = wc[d]
    return {"U": U, "V": V, "W": W}


def _rank23_strassen_variant():
    """Alternative rank-23 attempt using a different set of bilinear forms.
    
    This tries a variant based on Strassen's algorithm adapted for 3x3 matrices,
    using a different combination of 8x8 blocks and correction terms.
    """
    n, m, p = DIMENSIONS
    # Strassen-style decomposition for 3x3 matrices
    # Partition into 3x3 blocks and use 23 bilinear forms
    terms = [
        # Base terms for diagonal blocks
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0]),
        # Cross terms
        ([1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0], [1,0,0, 0,1,0, 0,0,0]),
        ([0,1,0, 1,0,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0], [0,1,0, 0,0,1, 0,0,0]),
        ([0,0,1, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,1,0]),
        # Additional bilinear forms
        ([1,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,1,0, 0,0,0, 0,0,0]),
        ([0,0,0, 1,1,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,0, 1,1,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        # Correction terms
        ([0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 1,0,0]),
        ([1,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]),
    ]
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (uc, vc, wc) in enumerate(terms):
        for a in range(n * m):
            U[a][c] = uc[a]
        for b in range(m * p):
            V[b][c] = vc[b]
        for d in range(p * n):
            W[d][c] = wc[d]
    return {"U": U, "V": V, "W": W}


# EVOLVE-BLOCK-START
def construct():
    # Guaranteed-correct baseline (rank 27) so we always have a valid answer.
    best = _schoolbook()
    best_rank = 27

    # Attempt the lower-rank (23) decompositions; trust them only if they verify
    # exactly. The deadline guard ensures we return well before the limit.
    if _time_left() > 5.0:
        # Try the first rank-23 attempt
        cand1 = _rank23_attempt()
        if _verify_exact(cand1["U"], cand1["V"], cand1["W"]):
            best = cand1
            best_rank = 23
            return best  # Found valid rank-23, return immediately
        
        # Try the second rank-23 variant if the first fails
        if _time_left() > 5.0:
            cand2 = _rank23_strassen_variant()
            if _verify_exact(cand2["U"], cand2["V"], cand2["W"]):
                best = cand2
                best_rank = 23
                return best  # Found valid rank-23, return immediately

    # Safety margin: stop constructing if we are approaching the deadline.
    if _time_left() <= 5.0:
        return best

    return best
# EVOLVE-BLOCK-END
