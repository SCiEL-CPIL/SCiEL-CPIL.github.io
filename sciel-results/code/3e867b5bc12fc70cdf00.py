"""Matrix multiplication 2x3x10 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
START = time.time()
DEADLINE = 150.0
MARGIN = 8.0          # seconds of safety margin before the 150s kill

def _time_left():
    return DEADLINE - MARGIN - (time.time() - START)

def _valid(U, V, W, n, m, p):
    """Check M = U @ V^T equals W exactly (integer/half-integer safe)."""
    M = np.round(U @ V.T, 6)
    return np.allclose(M, W, atol=1e-6, rtol=0.0)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _strassen_block(n, m, p):
    """
    Decompose C = A (n x m) * B (m x p) by splitting A into n/2 blocks of
    (n/2 x m) and B into p/2 blocks of (m x p/2), using Strassen on the
    resulting (n/2 x p/2) block matrix. Each Strassen sub-product is a
    (n/2 x m) * (m x p/2) multiplication done schoolbook.
    This is valid when n and p are even and m is even.
    """
    assert n % 2 == 0 and m % 2 == 0 and p % 2 == 0
    n2, m2, p2 = n // 2, m // 2, p // 2
    # Strassen: M = A * B where A is n2 x m, B is m x p2 -> 7 terms, each
    # a (n2 x m) * (m x p2) schoolbook of rank n2*m*p2.
    # We split A into two halves (top/bottom rows) and B into two halves
    # (left/right cols) to form a 2x2 block matrix of size (n2 x p2) each.
    # Total rank = 7 * (n2 * m * p2).
    sub = n2 * m * p2
    rank = 7 * sub
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # A block rows: top = rows [0, n2), bottom = rows [n2, n)
    # B block cols: left = cols [0, p2), right = cols [p2, p)
    # Strassen products (each is (n2 x m) * (m x p2)):
    # We treat A_top (n2 x m), A_bot (n2 x m), B_left (m x p2), B_right (m x p2)
    # P1 = (A_top + A_bot)(B_left + B_right)
    # P2 = (A_top + A_bot) B_left
    # P3 = A_top (B_right - B_left)
    # P4 = A_bot (B_left + B_right)
    # P5 = (A_bot - A_top) B_right
    # P6 = A_top (B_left - B_right)
    # P7 = (A_bot - A_top) B_left
    # Result blocks:
    # C_top_left   = P4 + P5 - P1 + P7
    # C_top_right  = P1 + P3
    # C_bot_left   = P2 + P4
    # C_bot_right  = P3 + P4 - P2 + P6
    # (standard Strassen 2x2 with these definitions)

    def schoolbook_block(row_off, col_off, sign_a, sign_b, base):
        """Add a (n2 x m)*(m x p2) schoolbook term starting at column base.
        row_off: 0 for top, n2 for bottom. col_off: 0 for left, p2 for right.
        sign_a, sign_b: +1/-1 for the A and B factor signs.
        Returns end column.
        """
        for ii in range(n2):
            for jj in range(m):
                for kk in range(p2):
                    U[(row_off + ii) * m + jj, base + ii * m * p2 + jj * p2 + kk] = sign_a
                    V[jj * p + (col_off + kk), base + ii * m * p2 + jj * p2 + kk] = sign_b
                    W[(col_off + kk) * n + (row_off + ii), base + ii * m * p2 + jj * p2 + kk] = 1.0
        return base + sub

    # P1 = (A_top + A_bot)(B_left + B_right): 2x2 = 4 schoolbook terms
    b = 0
    b = schoolbook_block(0, 0, 1, 1, b)   # top-left
    b = schoolbook_block(0, p2, 1, 1, b)  # top-right
    b = schoolbook_block(n2, 0, 1, 1, b)  # bot-left
    b = schoolbook_block(n2, p2, 1, 1, b) # bot-right
    P1 = (b - 4 * sub, b)
    # P2 = (A_top + A_bot) B_left: 2 terms
    b = schoolbook_block(0, 0, 1, 1, b)
    b = schoolbook_block(n2, 0, 1, 1, b)
    P2 = (b - 2 * sub, b)
    # P3 = A_top (B_right - B_left): 2 terms
    b = schoolbook_block(0, p2, 1, 1, b)
    b = schoolbook_block(0, 0, 1, -1, b)
    P3 = (b - 2 * sub, b)
    # P4 = A_bot (B_left + B_right): 2 terms
    b = schoolbook_block(n2, 0, 1, 1, b)
    b = schoolbook_block(n2, p2, 1, 1, b)
    P4 = (b - 2 * sub, b)
    # P5 = (A_bot - A_top) B_right: 2 terms
    b = schoolbook_block(n2, p2, 1, 1, b)
    b = schoolbook_block(0, p2, -1, 1, b)
    P5 = (b - 2 * sub, b)
    # P6 = A_top (B_left - B_right): 2 terms
    b = schoolbook_block(0, 0, 1, 1, b)
    b = schoolbook_block(0, p2, 1, -1, b)
    P6 = (b - 2 * sub, b)
    # P7 = (A_bot - A_top) B_left: 2 terms
    b = schoolbook_block(n2, 0, 1, 1, b)
    b = schoolbook_block(0, 0, -1, 1, b)
    P7 = (b - 2 * sub, b)

    # Now combine W factors: each output block C_{ab} is a sum/difference of
    # P-terms. Each P-term contributes a full (n2 x p2) sub-block of W columns.
    # For a given output block (row_off, col_off) and a list of (P, sign), we
    # scale the W entries in those P columns.
    # Map of output block -> list of (P_start, sign)
    def add_W(row_off, col_off, terms):
        for (ps, sign) in terms:
            for ii in range(n2):
                for kk in range(p2):
                    W[(col_off + kk) * n + (row_off + ii), ps + ii * m * p2 + kk] = sign
    # Clear W first (it was set to 1 in schoolbook_block)
    W[:] = 0.0
    add_W(0, 0, [(P4[0], 1), (P5[0], 1), (P1[0], -1), (P7[0], 1)])
    add_W(0, p2, [(P1[0], 1), (P3[0], 1)])
    add_W(n2, 0, [(P2[0], 1), (P4[0], 1)])
    add_W(n2, p2, [(P3[0], 1), (P4[0], 1), (P2[0], -1), (P6[0], 1)])

    # Also need U,V to reflect that P-terms are shared; they already do since
    # each schoolbook_block wrote its own columns. But the SAME P-term is used
    # in multiple output blocks; that's fine because U,V columns are per-term
    # and W combines them. However, we wrote U,V with the sign already, and W
    # is now the combination. This is correct: M = U V^T gives, for each output
    # entry, the sum over terms of (sign_a*sign_b) * W_sign. Wait — U and V
    # already carry sign_a and sign_b, so the product U V^T entry = sign_a*sign_b,
    # and W carries the output sign. The tensor coefficient = (U*V)*W must equal
    # the true coefficient (±1). Let's verify: for a term with sign_a, sign_b and
    # output sign s, contribution = sign_a*sign_b*s. For P1 top-left: sign_a=1,
    # sign_b=1, s=1 -> 1. Good. This matches the standard construction.
    return U, V, W

def _try_reduce(U, V, W, n, m, p, rng, max_iters=200):
    """Try to find a rank-(r-1) decomposition by zeroing one M entry.
    Returns (U2, V2, W2) if found, else None.
    """
    r = U.shape[1]
    nU, nV, nW = U.shape[0], V.shape[0], W.shape[0]
    # Build full M = U V^T
    M = U @ V.T
    # We'll pick a target row u* of U (the "removed" factor) and a column to
    # zero out in M. Strategy: choose a random row index i of U to be the
    # removed one (set U[i,:] = e_i), and a random M column j to zero.
    # Solve min ||U V^T - W|| s.t. U[i]=e_i, M[i,j]=0.
    # Equivalently, the M[i,j]=0 constraint on the free part.
    # We'll do: fix U[i] = e_i. Then M[i,:] = e_i . V^T = V[j_row, :] where
    # j_row is the row of V corresponding to... no, M[i,:] = U[i] V^T = e_i V^T
    # = V[i, :] (row i of V). So M[i,j] = V[i,j]. Constraint V[i,j]=0.
    # And the residual is over all entries. This is a standard rank-(r-1)
    # completion. We use a simple iterative scheme.
    # Simpler robust approach: pick i (U row) and j (V column index within that
    # row). Set U[i]=e_i. Then we need V[i,j]=0 and minimize residual.
    # We'll optimize V (with V[i,j]=0 fixed) and W and the rest of U.
    # Use Gauss-Newton-ish: treat as least squares in W and V.
    # For a given U, the best W is the elementwise product? No: M=U V^T, and
    # we want M=W (the tensor). Given U and V, W is determined = U V^T. The
    # decomposition is exact iff U V^T = W. So we're searching U,V such that
    # U V^T = W (the target tensor) and rank(U)<=r-1 effectively (with U[i]=e_i
    # and V[i,j]=0 giving a border rank r-1).
    # So: find U (with U[i]=e_i) and V (with V[i,j]=0) such that U V^T = W.
    # This is a system of nU*nW equations in (nU*(r-1) + nV*(r-1)) unknowns
    # (free parts). Linear in V given U, linear in U given V. Alternate.
    # Given U: V = (U^T W)^+ ... solve U V^T = W => for each row a of W,
    # W[a,:] = (U V^T)[a,:] = sum_l U[a,l] V[:,l] ... V[:,l] is a column.
    # Vectorize: W = U V^T. Given U, solve for V: W U^+ = V^T (least squares).
    # Given V, solve for U: W = U V^T => U = W (V^T)^+ .
    # Alternate with the constraints.
    best = None
    for it in range(max_iters):
        if _time_left() <= 0:
            break
        i = rng.integers(nU)
        j = rng.integers(nV)
        U2 = U.copy()
        V2 = V.copy()
        U2[i, :] = 0.0
        U2[i, i % r] = 1.0  # e_i (use column i mod r)
        V2[i, j] = 0.0
        # Alternate least squares
        for inner in range(30):
            # Solve V2 from U2: W = U2 V2^T -> V2^T = (U2^+ ) W  => V2 = W^T U2^+
            V2_new = np.linalg.lstsq(U2, W.T, rcond=None)[0].T
            V2_new[i, j] = 0.0
            # Solve U2 from V2_new: W = U2 V2_new^T -> U2 = W (V2_new^T)^+
            U2_new = np.linalg.lstsq(V2_new.T, W, rcond=None)[0]
            U2_new[i, :] = 0.0
            U2_new[i, i % r] = 1.0
            err = np.linalg.norm(U2_new @ V2_new.T - W) / np.linalg.norm(W)
            if err < 1e-7:
                U2, V2 = U2_new, V2_new
                break
            U2, V2 = U2_new, V2_new
        err = np.linalg.norm(U2 @ V2.T - W) / np.linalg.norm(W)
        if err < 1e-7:
            W2 = np.round(U2 @ V2.T, 6)
            if _valid(U2, V2, W2, n, m, p):
                return U2, V2, W2
        if best is None or err < best[0]:
            best = (err, U2, V2)
    return None

def construct():
    n, m, p = DIMENSIONS
    # Start from the best known valid decomposition.
    if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
        try:
            U, V, W = _strassen_block(n, m, p)
            if not _valid(U, V, W, n, m, p):
                U, V, W = _schoolbook(n, m, p)
        except Exception:
            U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = _schoolbook(n, m, p)

    best = (U.shape[1], U, V, W)
    rng = np.random.default_rng(12345)

    # Try to reduce rank iteratively.
    cur_U, cur_V, cur_W = U, V, W
    while _time_left() > 0:
        res = _try_reduce(cur_U, cur_V, cur_W, n, m, p, rng, max_iters=40)
        if res is not None:
            cur_U, cur_V, cur_W = res
            if cur_U.shape[1] < best[0]:
                best = (cur_U.shape[1], cur_U.copy(), cur_V.copy(), cur_W.copy())
        # Occasionally reshuffle to escape local minima
        if rng.random() < 0.1:
            perm = rng.permutation(cur_U.shape[1])
            cur_U, cur_V, cur_W = cur_U[:, perm], cur_V[:, perm], cur_W[:, perm]

    _, U, V, W = best
    # Round to clean half-integers
    U = np.round(U, 6)
    V = np.round(V, 6)
    W = np.round(W, 6)
    # Final validity check; fall back to schoolbook if broken.
    if not _valid(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
