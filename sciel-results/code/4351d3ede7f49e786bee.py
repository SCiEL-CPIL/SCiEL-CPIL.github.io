"""
Stronger approach for 4x4 * 4x9 matrix multiplication rank minimization.

Strategy:
1. Use a fast numpy-based validator that checks the full tensor identity.
2. Start with a good structured candidate (block-Strassen / Winograd-style blocks)
   to get a baseline well below 144.
3. Perform structured random local search: start from the best candidate, then
   randomly zero out columns and re-solve / adjust, or perform random column
   replacements, and validate quickly.
4. Use time-boxed search with a safety margin.

Key insight: The rank of 4x4*4x9 is at most 144 (schoolbook). We can do much better
by using Strassen on 2x2 blocks. For 4x4*4x9, we can split into blocks:
- A is 4x4, B is 4x9, C is 4x9.
- Split B into three 4x3 blocks? Or 4x4 + 4x4 + 4x1.
- For each 4x4 block of B, we can use a 4x4*4x4 multiplication. The best known rank
  for 4x4*4x4 is 49 (Strassen-based) or 49. Actually, the rank of 4x4*4x4 is 49
  (using Strassen recursively: 7^2 = 49).
- So for two 4x4 blocks of B, we'd use 49 + 49 = 98 terms, plus 16 for the 4x1 block = 114.
- But we can do better by sharing structure.

Actually, let's think more carefully. The matrix multiplication tensor for n x m * m x p
has rank at most n*m*p. For 4x4*4x9, that's 144.

Using Strassen:
- For 2x2*2x2, rank 7.
- For 4x4*4x4, we can use 2x2 blocks: 7 multiplications of 2x2, each of rank 7, so 49.
- For 4x4*4x9, we can split B into 4x4, 4x4, 4x1.
  - 4x4*4x4 part: rank 49
  - 4x4*4x4 part: rank 49
  - 4x4*4x1 part: rank 16 (schoolbook, since it's just 4x4*4x1)
  - Total: 49+49+16 = 114

But we can do better. The 4x4*4x1 part can be done with rank 16, but maybe we can
share some structure. Actually, no, because the output dimensions differ.

Let me try a different approach: use the fact that we only need the tensor identity.
I'll generate the 49-term Strassen decomposition for 4x4*4x4, then extend it to 4x4*4x9
by adding terms for the extra column.

Actually, the simplest strong candidate: use schoolbook for the 4x4*4x9 but with
Strassen on the 2x2 sub-blocks. Let me just implement a fast validator and do a
structured random search starting from a good initial point.
"""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]
DEADLINE = 140.0  # safety margin before 150s kill


def _validate_fast(U, V, W, n, m, p):
    """Fast validation using numpy. U: (n*m, r), V: (m*p, r), W: (p*n, r).
    Check that for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1.
    """
    r = U.shape[1]
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    # Reshape for efficient computation
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)

    result = np.zeros((n, m, p), dtype=np.float64)
    for c in range(r):
        uc = U3[:, :, c]  # (n, m)
        vc = V3[:, :, c]  # (m, p)
        wc = W3[:, :, c]  # (p, n)
        temp = uc[:, :, None] * vc[None, :, :]  # (n, m, p)
        wc_t = wc.T  # (n, p), so wc_t[i,k] = wc[k,i]
        result += temp * wc_t[:, None, :]  # (n, m, p) * (n, 1, p) -> (n, m, p)

    if not np.allclose(result, 1.0, atol=1e-9):
        return False
    return True


def _make_strassen_4x4_terms():
    """Return list of (i, j, k, coeff) for 4x4*4x4 Strassen decomposition (49 terms)."""
    def strassen_2x2():
        """Return list of (a_i, a_j, b_j, b_k, coeff) for 2x2*2x2 Strassen."""
        terms = []
        # M1 = (A00+A11)(B00+B11)
        terms += [(0,0,0,0,1), (0,0,1,1,1), (1,1,0,0,1), (1,1,1,1,1)]
        # M2 = A11*(B10-B00)
        terms += [(1,1,1,0,1), (1,1,0,0,-1)]
        # M3 = (A10+A11)*B01
        terms += [(1,0,0,1,1), (1,1,0,1,1)]
        # M4 = A00*(B01-B11)
        terms += [(0,0,0,1,1), (0,0,1,1,-1)]
        # M5 = (A00+A01)(B11+B00)
        terms += [(0,0,1,1,1), (0,0,0,0,1), (0,1,1,1,1), (0,1,0,0,1)]
        # M6 = (A01-A11)*(B10+B11)
        terms += [(0,1,1,0,1), (0,1,1,1,1), (1,1,1,0,-1), (1,1,1,1,-1)]
        # M7 = (A10-A00)*(B00+B01)
        terms += [(1,0,0,0,1), (1,0,0,1,1), (0,0,0,0,-1), (0,0,0,1,-1)]
        return terms

    s2 = strassen_2x2()

    block_defs = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),  # S1
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),           # S2
        ([(1,0,1), (1,1,1)], [(0,1,1)]),            # S3
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),           # S4
        ([(0,0,1), (0,1,1)], [(1,1,1), (0,0,1)]),   # S5
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),  # S6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),  # S7
    ]

    all_terms = []
    for A_blocks, B_blocks in block_defs:
        for (abr, abc, asgn) in A_blocks:
            for (bbr, bbc, bsgn) in B_blocks:
                for (ai, aj, bj, bk, coeff) in s2:
                    i = 2*abr + ai
                    j = 2*abc + aj
                    k = 2*bbc + bk
                    total_coeff = asgn * bsgn * coeff
                    if total_coeff != 0:
                        all_terms.append((i, j, k, total_coeff))

    return all_terms


def schoolbook_candidate():
    n, m, p = DIMENSIONS
    r = n * m * p
    U = np.zeros((n*m, r), dtype=np.float64)
    V = np.zeros((m*p, r), dtype=np.float64)
    W = np.zeros((p*n, r), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
    return U, V, W


def strassen_block_candidate():
    """Use Strassen 4x4 for the first 8 columns of B, schoolbook for last column."""
    n, m, p = DIMENSIONS
    terms_4x4 = _make_strassen_4x4_terms()

    all_terms = []
    # First 4x4 block of B (cols 0-3)
    for (i, j, k, coeff) in terms_4x4:
        all_terms.append((i, j, k, coeff))
    # Second 4x4 block of B (cols 4-7)
    for (i, j, k, coeff) in terms_4x4:
        k_new = k + 4
        all_terms.append((i, j, k_new, coeff))
    # Last column (col 8): schoolbook
    for i in range(n):
        for j in range(m):
            all_terms.append((i, j, 8, 1.0))

    r = len(all_terms)
    U = np.zeros((n*m, r), dtype=np.float64)
    V = np.zeros((m*p, r), dtype=np.float64)
    W = np.zeros((p*n, r), dtype=np.float64)
    for c, (i, j, k, coeff) in enumerate(all_terms):
        U[i*m+j, c] = coeff
        V[j*p+k, c] = 1.0
        W[k*n+i, c] = 1.0
    return U, V, W


def _try_remove_column(U, V, W, c, n, m, p):
    """Try removing column c. Return (success, U_new, V_new, W_new)."""
    U_test = np.delete(U, c, axis=1)
    V_test = np.delete(V, c, axis=1)
    W_test = np.delete(W, c, axis=1)
    if _validate_fast(U_test, V_test, W_test, n, m, p):
        return True, U_test, V_test, W_test
    return False, None, None


def _try_swap_columns(U, V, W, c1, c2, n, m, p):
    """Try swapping columns c1 and c2 (or zeroing one). Return (success, U_new, V_new, W_new)."""
    # Try zeroing c1
    U_test = U.copy()
    V_test = V.copy()
    W_test = W.copy()
    U_test[:, c1] = 0
    V_test[:, c1] = 0
    W_test[:, c1] = 0
    if _validate_fast(U_test, V_test, W_test, n, m, p):
        return True, U_test, V_test, W_test
    # Try zeroing c2
    U_test = U.copy()
    V_test = V.copy()
    W_test = W.copy()
    U_test[:, c2] = 0
    V_test[:, c2] = 0
    W_test[:, c2] = 0
    if _validate_fast(U_test, V_test, W_test, n, m, p):
        return True, U_test, V_test, W_test
    return False, None, None


def random_local_search(U0, V0, W0, n, m, p, max_time=100.0):
    """Local search: try to reduce rank by zeroing out columns and re-validating."""
    t0 = time.time()
    r0 = U0.shape[1]
    best_r = r0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    rng = random.Random(42)

    # First pass: try removing each column in order
    for c in range(best_r):
        if time.time() - t0 > max_time:
            break
        success, U_new, V_new, W_new = _try_remove_column(best_U, best_V, best_W, c, n, m, p)
        if success:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_r -= 1
            c = -1  # restart from beginning since indices shifted

    # Second pass: random removal attempts
    for iteration in range(5000):
        if time.time() - t0 > max_time:
            break
        if best_r <= 80:
            break
        c = rng.randint(0, best_r - 1)
        success, U_new, V_new, W_new = _try_remove_column(best_U, best_V, best_W, c, n, m, p)
        if success:
            best_U, best_V, best_W = U_new, V_new, W_new
            best_r -= 1

    return best_r, best_U, best_V, best_W


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS

    # Start with Strassen block candidate
    U, V, W = strassen_block_candidate()
    r = U.shape[1]

    if not _validate_fast(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = schoolbook_candidate()
        r = U.shape[1]

    # Local search to reduce rank
    remaining = DEADLINE - (time.time() - t0) - 5
    if remaining > 10:
        r, U, V, W = random_local_search(U, V, W, n, m, p, max_time=remaining)

    # Convert to list format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
