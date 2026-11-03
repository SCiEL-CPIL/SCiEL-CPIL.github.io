import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]


def _schoolbook(n, m, p):
    """Schoolbook (rank n*m*p) decomposition as lists."""
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


def _verify(U, V, W, n, m, p):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True


def _derive_W(U, V, n, m, p, tol=1e-9):
    """
    Given U (n*m x r) and V (m*p x r), solve for W (p*n x r) so that the
    decomposition is exact, i.e. for every output (i,k):
        sum_c W[k*n+i, c] * P_c[i,k] = 1
    where P_c[i,k] = sum_j U[i*m+j, c] * V[j*p+k, c].
    Returns W (numpy float) if the system is consistent, else None.
    """
    r = U.shape[1]
    P = np.zeros((r, n, p))
    for c in range(r):
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += U[i * m + j, c] * V[j * p + k, c]
                P[c, i, k] = s
    # Reshape to (r, n*p); W row for output (i,k) is index k*n+i.
    # We need, for each output o = k*n+i:  sum_c W[o,c] * P[c, i, k] = 1.
    P2 = P.reshape(r, n * p)  # column index = i*p + k
    # Build B of shape (n*p, r): B[o, c] = P[c, i, k] where o = k*n+i.
    B = np.zeros((n * p, r))
    for i in range(n):
        for k in range(p):
            o = k * n + i
            B[o, :] = P2[:, i * p + k]
    # Solve B @ w = 1 for each row o (independent systems, underdetermined).
    W = np.zeros((n * p, r))
    for o in range(n * p):
        row = B[o, :]
        if np.allclose(row, 0.0, atol=tol):
            return None  # cannot produce output o
        # Particular solution via lstsq (min-norm).
        sol, residuals, rank_, sv = np.linalg.lstsq(row, np.ones(1), rcond=None)
        # Check consistency: B[o] @ sol should be 1.
        if abs(np.dot(row, sol) - 1.0) > 1e-6:
            return None
        W[o, :] = sol
    return W


def _try_block(U, V, n, m, p, tol=1e-9):
    """Given U,V for an n x m x p block, derive W and verify. Returns (W, rank) or None."""
    W = _derive_W(U, V, n, m, p, tol)
    if W is None:
        return None
    # Verify exactness.
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-6:
                    return None
    return (W, r)


def _strassen_like_block(n, m, p, deadline, start_time, margin=20.0):
    """
    Search for a low-rank n x m x p decomposition using a Strassen-like
    construction: each factor vector is a sum/difference of a subset of
    coordinate unit vectors. Derive W exactly. Returns best (U,V,W) or None.
    """
    best = None
    best_rank = n * m * p + 1

    # Candidate subsets: for a vector of length L, generate all non-empty
    # subsets with +/- signs, but limit to subsets of size <= 3 to keep it tractable.
    # Coordinates for U: (i,j) in range(n*m); V: (j,k) in range(m*p); W: (k,i) in range(p*n).
    # We build U and V terms; W is derived.

    def gen_subsets(L, max_size):
        """Yield (indices, signs) for non-empty subsets of range(L) of size<=max_size."""
        # singletons
        for a in range(L):
            yield ((a,), (1,))
        if max_size >= 2:
            for a in range(L):
                for b in range(a + 1, L):
                    for sa in (1, -1):
                        for sb in (1, -1):
                            yield ((a, b), (sa, sb))
        if max_size >= 3:
            for a in range(L):
                for b in range(a + 1, L):
                    for c in range(b + 1, L):
                        for sa in (1, -1):
                            for sb in (1, -1):
                                for sc in (1, -1):
                                    yield ((a, b, c), (sa, sb, sc))

    def make_vec(L, indices, signs):
        v = np.zeros(L)
        for idx, s in zip(indices, signs):
            v[idx] = s
        return v

    # Precompute all candidate vector specs for U and V.
    # To limit runtime, cap the number of distinct U-terms and V-terms we try.
    U_specs = list(gen_subsets(n * m, 3))
    V_specs = list(gen_subsets(m * p, 3))

    # We will try combinations of U_specs and V_specs to form rank-r factorizations.
    # Strategy: try ranks r = 4 .. 12, and for each r, try r distinct (U,V) term pairs.
    # To keep it fast, we sample a bounded number of term pairs.
    max_terms = 14
    # Build a pool of term pairs (u_vec, v_vec) and try to assemble them.
    # We'll just try the first `max_terms` U_specs paired with the first `max_terms` V_specs
    # in a diagonal fashion, then also try larger ranks by taking more pairs.
    # Simpler: iterate over ranks and, for each rank, take the first `r` U_specs and
    # first `r` V_specs (diagonal pairing) and test.

    for r in range(4, max_terms + 1):
        if time.time() > start_time + (150 - margin):
            break
        # Diagonal pairing: term t uses U_specs[t] and V_specs[t].
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        for t in range(r):
            U[:, t] = make_vec(n * m, U_specs[t][0], U_specs[t][1])
            V[:, t] = make_vec(m * p, V_specs[t][0], V_specs[t][1])
        res = _try_block(U, V, n, m, p)
        if res is not None:
            W, rr = res
            if rr < best_rank:
                best_rank = rr
                best = (U, V, W)
    return best


def _bclr_known_block(n, m, p):
    """
    Attempt the known BCLR-style rank-23 3x3x3 decomposition by constructing
    U and V from sum/difference terms and deriving W. Returns (U,V,W) or None.
    This is the classic 23-product 3x3x3 scheme.
    """
    if not (n == 3 and m == 3 and p == 3):
        return None
    # A/B/C index layout: A[i*3+j], B[j*3+k], C[i*3+k]; W index = k*3+i.
    # We'll encode the 23 BCLR products as (U-vec, V-vec) pairs.
    # Standard BCLR 23-term products (U side, V side):
    terms = [
        # (u_indices, u_signs, v_indices, v_signs)
        ((0, 4, 8), (1, 1, 1), (0, 4, 8), (1, 1, 1)),   # (A00+A11+A22)(B00+B11+B22)
        ((0, 4), (1, 1), (0, 4), (1, 1)),                # (A00+A11)(B00+B11)
        ((0, 8), (1, 1), (0, 8), (1, 1)),                # (A00+A22)(B00+B22)
        ((4, 8), (1, 1), (4, 8), (1, 1)),                # (A11+A22)(B11+B22)
        ((0, 5, 7), (1, 1, 1), (0, 5, 7), (1, 1, 1)),    # (A00+A12+A21)(B00+B12+B21)
        ((0, 5), (1, 1), (0, 5), (1, 1)),                # (A00+A12)(B00+B12)
        ((0, 7), (1, 1), (0, 7), (1, 1)),                # (A00+A21)(B00+B21)
        ((5, 7), (1, 1), (5, 7), (1, 1)),                # (A12+A21)(B12+B21)
        ((4, 5, 6), (1, 1, 1), (4, 5, 6), (1, 1, 1)),    # (A11+A12+A20)(B11+B12+B20)
        ((4, 5), (1, 1), (4, 5), (1, 1)),                # (A11+A12)(B11+B12)
        ((4, 6), (1, 1), (4, 6), (1, 1)),                # (A11+A20)(B11+B20)
        ((5, 6), (1, 1), (5, 6), (1, 1)),                # (A12+A20)(B12+B20)
        ((8, 7, 3), (1, 1, 1), (8, 7, 3), (1, 1, 1)),    # (A22+A21+A10)(B22+B21+B10)
        ((8, 7), (1, 1), (8, 7), (1, 1)),                # (A22+A21)(B22+B21)
        ((8, 3), (1, 1), (8, 3), (1, 1)),                # (A22+A10)(B22+B10)
        ((7, 3), (1, 1), (7, 3), (1, 1)),                # (A21+A10)(B21+B10)
        ((0,), (1,), (0,), (1,)),                        # A00*B00
        ((4,), (1,), (4,), (1,)),                        # A11*B11
        ((8,), (1,), (8,), (1,)),                        # A22*B22
        ((1,), (1,), (3,), (1,)),                        # A01*B10
        ((2,), (1,), (6,), (1,)),                        # A02*B20
        ((3,), (1,), (1,), (1,)),                        # A10*B01
        ((6,), (1,), (2,), (1,)),                        # A20*B02
    ]
    r = len(terms)
    U = np.zeros((9, r))
    V = np.zeros((9, r))
    for t, (ui, us, vi, vs) in enumerate(terms):
        for idx, s in zip(ui, us):
            U[idx, t] += s
        for idx, s in zip(vi, vs):
            V[idx, t] += s
    res = _try_block(U, V, 3, 3, 3)
    if res is None:
        return None
    W, rr = res
    return (U, V, W)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before the 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook fallback (always valid).
    sb = _schoolbook(n, m, p)

    # Best result found so far (start with schoolbook).
    best_U, best_V, best_W = (np.array(sb["U"], dtype=float),
                              np.array(sb["V"], dtype=float),
                              np.array(sb["W"], dtype=float))
    best_rank = best_U.shape[1]

    # --- Step 1: try a known 3x3x3 block decomposition (rank 23) ---
    block = None
    # Try the known BCLR-style block first (fast, deterministic).
    bclr = _bclr_known_block(3, 3, 3)
    if bclr is not None:
        U3, V3, W3 = bclr
        if _verify(U3.tolist(), V3.tolist(), W3.tolist(), 3, 3, 3):
            block = (U3, V3, W3)

    # --- Step 2: if no known block, search for a low-rank 3x3x3 block ---
    if block is None and time.time() < deadline - 30:
        block = _strassen_like_block(3, 3, 3, deadline, start_time)

    # --- Step 3: assemble 3x3x6 from two 3x3x3 blocks (direct sum) ---
    if block is not None:
        U3, V3, W3 = block
        r3 = U3.shape[1]
        rank46 = 2 * r3
        U46 = np.zeros((9, rank46))
        V46 = np.zeros((18, rank46))
        W46 = np.zeros((18, rank46))
        # First half of B (cols 0..2) -> first half of C (cols 0..2).
        U46[:, 0:r3] = U3
        V46[0:9, 0:r3] = V3
        W46[0:9, 0:r3] = W3
        # Second half of B (cols 3..5) -> second half of C (cols 3..5).
        U46[:, r3:2 * r3] = U3
        V46[9:18, r3:2 * r3] = V3
        W46[9:18, r3:2 * r3] = W3

        if _verify(U46.tolist(), V46.tolist(), W46.tolist(), 3, 3, 6):
            if rank46 < best_rank:
                best_U, best_V, best_W = U46, V46, W46
                best_rank = rank46

    # --- Step 4: if still at schoolbook and time remains, try a direct
    #     low-rank 3x3x6 search (Strassen-like) as a last resort. ---
    if best_rank >= n * m * p and time.time() < deadline - 20:
        direct = _strassen_like_block(3, 3, 6, deadline, start_time)
        if direct is not None:
            Ud, Vd, Wd = direct
            if _verify(Ud.tolist(), Vd.tolist(), Wd.tolist(), 3, 3, 6):
                rd = Ud.shape[1]
                if rd < best_rank:
                    best_U, best_V, best_W = Ud, Vd, Wd
                    best_rank = rd

    # Return the best valid result as lists.
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
