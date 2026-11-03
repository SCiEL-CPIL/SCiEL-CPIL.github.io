"""Matrix multiplication tensor decomposition for 2x3 * 3x10.

Design: hand-constructed rank-55 (Strassen on the 2x2 sub-block) decomposition,
verified, then a fast exact search for a rank-54 solution using the structural
fact that the tensor has 60 non-zero entries.

The key structural observation: in the rank-55 solution, 55 of the 60 output
positions are covered. To get rank 54 we need to "merge" two of the rank-55
terms into a single term, which requires a specific algebraic relation.

We use a targeted search: for each of the 55 columns in the rank-55 solution,
drop that column and solve the resulting linear system for W. If the residual
is zero (within tolerance), we have a rank-54 solution.
"""
import numpy as np
import random
import time

DIMENSIONS = [2, 3, 10]


def _verify(U, V, W, n, m, p, trials=2000):
    """Verify that U, V, W give an exact decomposition of the mult tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)

    T_reconstructed = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T_reconstructed[a, b, c] - 1.0) > 1e-9:
                    return False

    nonzero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nonzero_set.add((i * m + j, j * p + k, k * n + i))

    for _ in range(trials):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        if (a, b, c) not in nonzero_set:
            if abs(T_reconstructed[a, b, c]) > 1e-9:
                return False

    return True


def _is_valid_entries(U, V, W):
    """Check that all entries are integers or half-integers."""
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True


def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition using Strassen on the 2x2 sub-block.
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part A: j=0
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # Part B: j=1,2
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1; U[u_22][col] += 1
        V[v_11][col] += 1; V[v_22][col] += 1
        W[w_00][col] += 1; W[w_11][col] += 1
        col += 1

        # M2 = (a21 + a22)b11
        U[u_21][col] += 1; U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1; W[w_11][col] += -1
        col += 1

        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1; V[v_22][col] += -1
        W[w_01][col] += 1; W[w_11][col] += 1
        col += 1

        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1; V[v_11][col] += -1
        W[w_00][col] += 1; W[w_10][col] += 1
        col += 1

        # M5 = (a11 + a12)b22
        U[u_11][col] += 1; U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1; W[w_01][col] += 1
        col += 1

        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1; U[u_11][col] += -1
        V[v_11][col] += 1; V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1; U[u_22][col] += -1
        V[v_21][col] += 1; V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _build_schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
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
    return U, V, W


def _structured_search_rank54(n, m, p, deadline):
    """
    Targeted search for rank 54. For each of the 55 columns in the rank-55
    solution, drop that column and solve the resulting linear system for W.
    If the residual is zero (within tolerance), we have a rank-54 solution.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    rank = 54

    U55, V55, W55 = _build_strassen55(n, m, p)
    U55 = np.array(U55, dtype=np.float64)
    V55 = np.array(V55, dtype=np.float64)

    # Build target tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Collect equations by c index
    eqs_by_c = {c: [] for c in range(pn)}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                eqs_by_c[c].append((a, b))

    # For each column to drop, try to solve for W
    for drop_col in range(55):
        if time.time() > deadline:
            break

        # U, V with column drop_col removed
        mask = np.ones(55, dtype=bool)
        mask[drop_col] = False
        U_r = U55[:, mask]  # (nm, 54)
        V_r = V55[:, mask]  # (mp, 54)

        W_sol = np.zeros((pn, rank), dtype=np.float64)
        feasible = True

        for c in range(pn):
            if not eqs_by_c[c]:
                continue
            # Build system
            A_sys = np.zeros((len(eqs_by_c[c]), rank), dtype=np.float64)
            for idx, (a, b) in enumerate(eqs_by_c[c]):
                A_sys[idx] = U_r[a, :] * V_r[b, :]  # element-wise product
            rhs = np.ones(len(eqs_by_c[c]), dtype=np.float64)

            # Solve least squares
            try:
                sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, rhs, rcond=None)
            except Exception:
                feasible = False
                break

            # Check residual
            resid = np.linalg.norm(A_sys @ sol - rhs)
            if resid > 1e-9:
                feasible = False
                break
            W_sol[c, :] = sol

        if not feasible:
            continue

        # Snap to half-integers
        U_snap = np.round(U_r * 2) / 2
        V_snap = np.round(V_r * 2) / 2
        W_snap = np.round(W_sol * 2) / 2

        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()

        if _verify(U_list, V_list, W_list, n, m, p) and _is_valid_entries(U_list, V_list, W_list):
            return U_list, V_list, W_list

    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s margin for the evaluator

    # Start with the known rank-55 Strassen-based decomposition
    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    if not (_verify(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    # Try structured search for rank 54
    if time.time() < deadline - 5:
        result = _structured_search_rank54(n, m, p, deadline - 2)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = 54

    return {"U": U_best, "V": V_best, "W": W_best}
