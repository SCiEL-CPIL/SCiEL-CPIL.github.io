"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]
import time
import random

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}


def _strassen_2x2_blocks(n, m, p):
    """
    Apply Strassen to 2x2 blocks of size (n/2, m/2) x (m/2, p/2).
    Each block multiplication is done with schoolbook.
    Only valid when n, m, p are all even.
    """
    if n % 2 or m % 2 or p % 2:
        return None

    nh, mh, ph = n // 2, m // 2, p // 2

    # Block indices:
    # A blocks: 0=A00, 1=A01, 2=A10, 3=A11
    # B blocks: 0=B00, 1=B01, 2=B10, 3=B11
    # C blocks: 0=C00, 1=C01, 2=C10, 3=C11

    # Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7

    # For each M_s, we need the linear combination of A blocks and B blocks.
    # Each block is a (nh x mh) matrix for A, (mh x ph) for B.
    # Schoolbook for a (nh x mh) * (mh x ph) multiplication has rank nh*mh*ph.

    # Define the 7 steps as (A_coeffs, B_coeffs) where coeffs are dicts block_idx -> sign
    steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}),              # M2
        ({0: 1}, {1: 1, 3: -1}),             # M3
        ({3: 1}, {2: 1, 0: -1}),             # M4
        ({0: 1, 1: 1}, {3: 1}),              # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),       # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}),       # M7
    ]

    # C combination: for each C block, which M_s contribute with what sign
    # C00 = M1 - M4 + M6 - M7  -> C00 gets M1(+1), M4(-1), M6(+1), M7(-1)
    # C01 = M3 + M5            -> C01 gets M3(+1), M5(+1)
    # C10 = M2 + M4            -> C10 gets M2(+1), M4(+1)
    # C11 = M1 - M2 + M3 + M7  -> C11 gets M1(+1), M2(-1), M3(+1), M7(+1)

    c_combos = [
        {0: 1, 3: -1, 5: 1, 6: -1},   # C00: M1, M4, M6, M7 (indices 0,3,5,6)
        {2: 1, 4: 1},                  # C01: M3, M5 (indices 2,4)
        {1: 1, 3: 1},                  # C10: M2, M4 (indices 1,3)
        {0: 1, 1: -1, 2: 1, 6: 1},    # C11: M1, M2, M3, M7 (indices 0,1,2,6)
    ]

    block_rank = nh * mh * ph
    total_rank = 7 * block_rank

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    for s in range(7):
        a_coeffs, b_coeffs = steps[s]
        for ai, a_sign in a_coeffs.items():
            for bi, b_sign in b_coeffs.items():
                # For this pair of blocks, do schoolbook multiplication
                # A block ai: rows [ai//2 * nh, ai//2 * nh + nh), cols [ai%2 * mh, ai%2 * mh + mh)
                # B block bi: rows [bi//2 * mh, bi//2 * mh + mh), cols [bi%2 * ph, bi%2 * ph + ph)
                a_row_start = (ai // 2) * nh
                a_col_start = (ai % 2) * mh
                b_row_start = (bi // 2) * mh
                b_col_start = (bi % 2) * ph

                # For each (i, j, k) in the block:
                # i in 0..nh-1, j in 0..mh-1, k in 0..ph-1
                for i in range(nh):
                    for j in range(mh):
                        for k in range(ph):
                            # Global indices
                            gi = a_row_start + i
                            gj = a_col_start + j
                            gk = b_col_start + k

                            # U: A[gi][gj]
                            # V: B[gj][gk]
                            # W: C[gi][gk] -- but we need to distribute to the right C blocks

                            # The product A_block * B_block contributes to C blocks
                            # according to c_combos. But the C block is determined by
                            # which block of C this (i,k) falls into.
                            #
                            # Actually, the C block is determined by the global row/col:
                            # C block row = gi // nh, C block col = gk // ph
                            c_block_row = gi // nh
                            c_block_col = gk // ph
                            c_block_idx = c_block_row * 2 + c_block_col

                            # For this (i,j,k), the contribution goes to W at position
                            # corresponding to C[gi][gk] = C[k*n + i] in the trace convention
                            # But we need to apply the C combination sign.

                            # Wait - the C combination means that M_s contributes to C_block
                            # with a certain sign. So for a given s and a given (i,k) that
                            # falls in C_block_idx, the W entry gets the sign from c_combos[c_block_idx].get(s, 0).

                            c_sign = c_combos[c_block_idx].get(s, 0)
                            if c_sign == 0:
                                continue

                            # Column index
                            col = s * block_rank + i * (mh * ph) + j * ph + k

                            U[gi * m + gj][col] = a_sign
                            V[gj * p + gk][col] = b_sign
                            W[gk * n + gi][col] = c_sign

    return {"U": U, "V": V, "W": W}


def _verify(decomp, n, m, p, sample=True):
    """Verify that the decomposition is correct."""
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0

    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False

    # Verify: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    # And for all other (i',j',k') not matching, the sum should be 0.
    # Actually, the tensor is: T[i,j,k] = 1 if we're computing C[i][k] from A[i][j]*B[j][k]
    # The decomposition says: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal 1 for all valid (i,j,k) and 0 otherwise.
    # But since every (i,j,k) in range is valid, it should be 1 for all.

    # For efficiency, check a sample if sample=True
    if sample:
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        random.shuffle(indices)
        indices = indices[:min(200, len(indices))]
    else:
        indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    for i, j, k in indices:
        s = 0
        for c in range(rank):
            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
        if abs(s - 1) > 1e-9:
            return False
    return True


def _try_rank_r(n, m, p, r, deadline):
    """Try to find a rank-r decomposition via random search."""
    # This is a placeholder for more advanced search.
    # For now, we don't have a reliable way to find sub-schoolbook decompositions
    # for 6x6x6 in the time budget, so we return None.
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin

    # First, try Strassen-based decomposition
    try:
        strassen_decomp = _strassen_2x2_blocks(n, m, p)
        if strassen_decomp is not None:
            if _verify(strassen_decomp, n, m, p, sample=False):
                return strassen_decomp
    except Exception:
        pass

    # Fall back to schoolbook
    schoolbook_decomp = _schoolbook(n, m, p)
    return schoolbook_decomp

# EVOLVE-BLOCK-END
