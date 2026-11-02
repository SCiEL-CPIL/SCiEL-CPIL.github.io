"""Combined program: block Strassen + GF(2) support search + rational lift."""
import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 4, 4]
N, M, P = 3, 4, 4

# EVOLVE-BLOCK-START

def _verify_einsum(U, V, W, n=N, m=M, p=P):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(recon, np.ones((n, m, p)), atol=1e-9)


def _verify_strict(U, V, W, n=N, m=M, p=P, tol=1e-12):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(recon, np.ones((n, m, p)), atol=tol)


def schoolbook(na, mb, pb):
    rank = na * mb * pb
    U = [[0] * rank for _ in range(na * mb)]
    V = [[0] * rank for _ in range(mb * pb)]
    W = [[0] * rank for _ in range(pb * na)]
    col = 0
    for i in range(na):
        for j in range(mb):
            for k in range(pb):
                U[i*mb+j][col] = 1
                V[j*pb+k][col] = 1
                W[k*na+i][col] = 1
                col += 1
    return U, V, W


def strassen_2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    U[1][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W


def _add_sub(U, V, W, U_sub, V_sub, W_sub, i_off, j_off, k_off, na, mb, pb, col_start, n, m, p):
    r = len(U_sub[0])
    for c in range(r):
        col = col_start + c
        for i_prime in range(na):
            for j_prime in range(mb):
                u_val = U_sub[i_prime*mb + j_prime][c]
                if u_val != 0:
                    U[(i_off + i_prime)*m + (j_off + j_prime)][col] += u_val
        for j_prime in range(mb):
            for k_prime in range(pb):
                v_val = V_sub[j_prime*pb + k_prime][c]
                if v_val != 0:
                    V[(j_off + j_prime)*p + (k_off + k_prime)][col] += v_val
        for k_prime in range(pb):
            for i_prime in range(na):
                w_val = W_sub[k_prime*na + i_prime][c]
                if w_val != 0:
                    W[(k_off + k_prime)*n + (i_off + i_prime)][col] += w_val


def _build_2plus1_split():
    n, m, p = N, M, P
    rank_total = 4 * 7 + 4 * 4
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    col = 0
    for k_off in [0, 2]:
        for j_off in [0, 2]:
            Us, Vs, Ws = strassen_2x2()
            _add_sub(U, V, W, Us, Vs, Ws, 0, j_off, k_off, 2, 2, 2, col, n, m, p)
            col += 7
            Us, Vs, Ws = schoolbook(1, 2, 2)
            _add_sub(U, V, W, Us, Vs, Ws, 2, j_off, k_off, 1, 2, 2, col, n, m, p)
            col += 4
    return U, V, W


def _try_rank_reduction_via_merging(U, V, W, n, m, p, max_time=30):
    start = time.time()
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    if r > 48:
        return None
    best_rank = r
    for col_idx in range(r - 1, -1, -1):
        if time.time() - start > max_time:
            break
        U_test = np.delete(U, col_idx, axis=1)
        V_test = np.delete(V, col_idx, axis=1)
        W_test = np.delete(W, col_idx, axis=1)
        if _verify_einsum(U_test, V_test, W_test, n, m, p):
            U, V, W = U_test, V_test, W_test
            best_rank = W.shape[1]
    if best_rank < r:
        return U.tolist(), V.tolist(), W.tolist()
    return None


def _support_to_pattern(U_bin, V_bin, W_bin, n, m, p):
    """Extract support sets for each column from binary matrices."""
    r = U_bin.shape[1]
    supports = []
    for c in range(r):
        u_idx = set(np.nonzero(U_bin[:, c])[0])
        v_idx = set(np.nonzero(V_bin[:, c])[0])
        w_idx = set(np.nonzero(W_bin[:, c])[0])
        supports.append((u_idx, v_idx, w_idx))
    return supports


def _enumerate_supports(n, m, p, max_cols=16, deadline=None):
    """
    Enumerate candidate support patterns. A support for one column is a triple
    (S_U, S_V, S_W) of subsets of {0..n*m-1}, {0..m*p-1}, {0..p*n-1}.
    The column contributes to tensor entry (i,j,k) iff
        i*m+j in S_U AND j*p+k in S_V AND k*n+i in S_W.
    We need the multiset of columns to cover each (i,j,k) exactly once.
    """
    n, m, p = n, m, p
    nm, mp, pn = n*m, m*p, p*n
    # All (i,j,k) triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    # For each possible support triple, compute which (i,j,k) it covers
    # To keep it tractable, only consider "small" supports.
    # A natural small support: for a given (i,j,k), the minimal support is
    #   S_U = {i*m+j}, S_V = {j*p+k}, S_W = {k*n+i}
    # which covers exactly that triple. That's schoolbook.
    # To get rank < n*m*p, we need columns whose supports cover multiple triples.

    # Strategy: generate candidate columns as unions of minimal supports.
    # For each subset of triples of size 2, 3, 4, build the union support
    # and check if it's "consistent" (i.e., the union support covers
    # exactly those triples and no others).

    candidates = []
    # Singletons (schoolbook columns)
    for (i, j, k) in triples:
        s_u = {i*m+j}
        s_v = {j*p+k}
        s_w = {k*n+i}
        # Verify it covers only (i,j,k)
        covered = set()
        for (i2, j2, k2) in triples:
            if i2*m+j2 in s_u and j2*p+k2 in s_v and k2*n+i2 in s_w:
                covered.add((i2, j2, k2))
        if covered == {(i, j, k)}:
            candidates.append((frozenset(s_u), frozenset(s_v), frozenset(s_w), {(i, j, k)}))

    # Pairs of triples: try union of two minimal supports
    for idx in range(len(triples)):
        for idx2 in range(idx+1, len(triples)):
            i1, j1, k1 = triples[idx]
            i2, j2, k2 = triples[idx2]
            s_u = {i1*m+j1, i2*m+j2}
            s_v = {j1*p+k1, j2*p+k2}
            s_w = {k1*n+i1, k2*n+i2}
            covered = set()
            for (i3, j3, k3) in triples:
                if i3*m+j3 in s_u and j3*p+k3 in s_v and k3*n+i3 in s_w:
                    covered.add((i3, j3, k3))
            if covered == {(i1, j1, k1), (i2, j2, k2)}:
                candidates.append((frozenset(s_u), frozenset(s_v), frozenset(s_w), covered))
            if deadline and time.time() > deadline:
                return candidates

    # Triples of triples
    for idx in range(len(triples)):
        for idx2 in range(idx+1, len(triples)):
            for idx3 in range(idx2+1, len(triples)):
                i1, j1, k1 = triples[idx]
                i2, j2, k2 = triples[idx2]
                i3, j3, k3 = triples[idx3]
                s_u = {i1*m+j1, i2*m+j2, i3*m+j3}
                s_v = {j1*p+k1, j2*p+k2, j3*p+k3}
                s_w = {k1*n+i1, k2*n+i2, k3*n+i3}
                covered = set()
                for (i4, j4, k4) in triples:
                    if i4*m+j4 in s_u and j4*p+k4 in s_v and k4*n+i4 in s_w:
                        covered.add((i4, j4, k4))
                if covered == {(i1, j1, k1), (i2, j2, k2), (i3, j3, k3)}:
                    candidates.append((frozenset(s_u), frozenset(s_v), frozenset(s_w), covered))
                if deadline and time.time() > deadline:
                    return candidates

    return candidates


def _exact_cover_search(candidates, triples, max_rank, deadline):
    """
    Find a set of candidate columns whose covered-sets partition the triples.
    Use backtracking with pruning.
    """
    triples_set = set(triples)
    n_triples = len(triples)
    # Index candidates by which triples they cover
    # For efficiency, sort candidates by number of triples covered (descending)
    candidates_sorted = sorted(candidates, key=lambda c: -len(c[3]))

    # Precompute: for each triple, which candidates cover it
    triple_to_cands = {t: [] for t in triples}
    for ci, (su, sv, sw, covered) in enumerate(candidates_sorted):
        for t in covered:
            triple_to_cands[t].append(ci)

    best = None
    best_rank = max_rank + 1

    def backtrack(remaining, chosen, start_idx):
        nonlocal best, best_rank
        if time.time() > deadline:
            return
        if not remaining:
            rank = len(chosen)
            if rank < best_rank:
                best_rank = rank
                best = list(chosen)
            return
        if len(chosen) >= best_rank:
            return
        # Pick the triple with fewest candidate options
        target = min(remaining, key=lambda t: len([ci for ci in triple_to_cands[t] if ci >= start_idx]))
        for ci in triple_to_cands[target]:
            if ci < start_idx:
                continue
            su, sv, sw, covered = candidates_sorted[ci]
            if not covered.issubset(remaining):
                continue
            new_remaining = remaining - covered
            backtrack(new_remaining, chosen + [ci], ci + 1)

    backtrack(frozenset(triples_set), [], 0)
    return best, best_rank


def _lift_supports_to_coefficients(support_indices, candidates, n, m, p, deadline):
    """
    Given a list of candidate indices (each with a support), solve for
    rational coefficients u_c, v_c, w_c for each column c such that
    for all (i,j,k): sum_{c: (i,j,k) in covered_c} u_c * v_c * w_c = 1.

    Since we're on rationals, set one variable per column to 1 and solve.
    Actually, the scaling invariance means we can fix, say, w_c = 1 for each c,
    then u_c * v_c must sum to 1 over columns covering (i,j,k).

    This is still nonlinear. Instead, use a linearization:
    For each column c, let x_c = u_c, y_c = v_c, z_c = w_c.
    Constraint: for each (i,j,k), sum_{c in C(i,j,k)} x_c * y_c * z_c = 1.

    This is a system of polynomial equations. For small systems, we can try
    to solve by fixing z_c = 1 and then x_c * y_c are the unknowns, but
    then x_c * y_c is a single variable per column, say t_c.
    Then: for each (i,j,k), sum_{c in C(i,j,k)} t_c = 1.
    This is a linear system in t_c! And then we can set x_c = t_c, y_c = 1, z_c = 1.

    But wait: the support constraint says that for column c, only entries
    in S_U_c, S_V_c, S_W_c can be non-zero. If we set y_c = 1 for all entries
    in S_V_c, that's fine as long as S_V_c is non-empty. Similarly for others.

    So the approach:
    1. For the chosen supports, build the linear system:
       For each (i,j,k): sum_{c: (i,j,k) in covered_c} t_c = 1
    2. Solve for t_c (rational).
    3. Set U[i*m+j, c] = t_c if i*m+j in S_U_c else 0
       V[j*p+k, c] = 1 if j*p+k in S_V_c else 0
       W[k*n+i, c] = 1 if k*n+i in S_W_c else 0
    4. Verify.
    """
    n, m, p = n, m, p
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    r = len(support_indices)

    # Build linear system A * t = 1
    # A[i,j,k, c] = 1 if (i,j,k) in covered_c else 0
    # Use Fraction for exact arithmetic
    from fractions import Fraction
    A = [[Fraction(0)] * r for _ in range(len(triples))]
    for ci, c_idx in enumerate(support_indices):
        su, sv, sw, covered = candidates[c_idx]
        for (i, j, k) in covered:
            tri_idx = i * m * p + j * p + k
            A[tri_idx][ci] = Fraction(1)

    b = [Fraction(1)] * len(triples)

    # Solve using Gaussian elimination over Q
    def solve_linear(A, b):
        n_eq = len(A)
        n_var = len(A[0]) if n_eq > 0 else 0
        # Augmented matrix
        M = [row[:] + [b[i]] for i, row in enumerate(A)]
        n_rows = len(M)
        n_cols = len(M[0]) - 1
        rank = 0
        for col in range(n_cols):
            if time.time() > deadline:
                return None
            # Find pivot
            pivot = None
            for row in range(rank, n_rows):
                if M[row][col] != 0:
                    pivot = row
                    break
            if pivot is None:
                continue
            # Swap
            M[rank], M[pivot] = M[pivot], M[rank]
            # Eliminate
            pivot_val = M[rank][col]
            for row in range(n_rows):
                if row != rank and M[row][col] != 0:
                    factor = M[row][col] / pivot_val
                    for c2 in range(col, n_cols + 1):
                        M[row][c2] -= factor * M[rank][c2]
            rank += 1
        # Check consistency
        for row in range(rank, n_rows):
            if M[row][n_cols] != 0:
                return None  # Inconsistent
        # Extract solution (free variables set to 0)
        sol = [Fraction(0)] * n_cols
        for row in range(rank):
            # Find leading column
            for col in range(n_cols):
                if M[row][col] != 0:
                    sol[col] = M[row][n_cols] / M[row][col]
                    break
        return sol

    t_sol = solve_linear(A, b)
    if t_sol is None:
        return None

    # Build U, V, W
    U = [[Fraction(0)] * r for _ in range(n * m)]
    V = [[Fraction(0)] * r for _ in range(m * p)]
    W = [[Fraction(0)] * r for _ in range(p * n)]

    for ci, c_idx in enumerate(support_indices):
        su, sv, sw, covered = candidates[c_idx]
        tc = t_sol[ci]
        for u_idx in su:
            U[u_idx][ci] = tc
        for v_idx in sv:
            V[v_idx][ci] = Fraction(1)
        for w_idx in sw:
            W[w_idx][ci] = Fraction(1)

    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]

    return U_list, V_list, W_list


def _try_gf2_support_search(max_time=60):
    """
    Try to find a low-rank decomposition via support enumeration + linear lift.
    """
    start = time.time()
    deadline = start + max_time
    n, m, p = N, M, P

    # Enumerate candidate supports
    candidates = _enumerate_supports(n, m, p, deadline=deadline - 5)
    if time.time() > deadline - 5:
        return None

    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]

    # Try to find an exact cover with rank < 44
    # Start from rank 24 upward
    for target_rank in range(24, 44):
        if time.time() > deadline - 10:
            break
        # Use the exact cover search
        best, best_rank = _exact_cover_search(candidates, triples, target_rank, deadline - 10)
        if best is not None and best_rank < 44:
            # Lift to coefficients
            result = _lift_supports_to_coefficients(best, candidates, n, m, p, deadline)
            if result is not None:
                U, V, W = result
                if _verify_einsum(U, V, W, n, m, p):
                    return U, V, W, best_rank
    return None


def construct():
    n, m, p = N, M, P
    start_time = time.time()
    deadline = start_time + 110

    # Build the 2+1 row split decomposition (rank 44)
    U, V, W = _build_2plus1_split()

    best_rank = 44
    best_result = (U, V, W)

    # Verify
    if _verify_einsum(U, V, W, n, m, p):
        # Try GF(2) support search for lower rank
        if time.time() < deadline - 30:
            gf2_result = _try_gf2_support_search(max_time=min(60, deadline - time.time() - 15))
            if gf2_result is not None:
                U_gf2, V_gf2, W_gf2, rank_gf2 = gf2_result
                if _verify_strict(U_gf2, V_gf2, W_gf2, n, m, p):
                    best_rank = rank_gf2
                    best_result = (U_gf2, V_gf2, W_gf2)

        # Try rank reduction via merging
        if time.time() < deadline - 15:
            reduced = _try_rank_reduction_via_merging(best_result[0], best_result[1], best_result[2], n, m, p, max_time=15)
            if reduced is not None:
                U_r, V_r, W_r = reduced
                if _verify_einsum(U_r, V_r, W_r, n, m, p):
                    new_rank = len(U_r[0])
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best_result = (U_r, V_r, W_r)

        # Strict verification
        if _verify_strict(best_result[0], best_result[1], best_result[2], n, m, p, tol=1e-12):
            return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}

    # Fallback: schoolbook rank 48
    Us, Vs, Ws = schoolbook(n, m, p)
    if _verify_einsum(Us, Vs, Ws, n, m, p):
        return {"U": Us, "V": Vs, "W": Ws}

    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}

# EVOLVE-BLOCK-END
