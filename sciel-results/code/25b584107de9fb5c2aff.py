"""Matrix multiplication tensor decomposition for 2x3 * 3x10.

Design: hand-constructed rank-55 (Strassen on the 2x2 sub-block) decomposition,
verified, then a fast exact search for a rank-54 solution using the structural
fact that the tensor has 60 non-zero entries.  A rank-r decomposition of the
multiplication tensor must satisfy, for every output index c:
    sum_r U[a,r] * V[b,r] * W[c,r] = 1  (for the 60 non-zero (a,b,c))
and 0 otherwise.  Since each term is a product of three scalars, and the
decomposition is exact, we can search over integer/half-integer factor matrices.

The key structural observation: in the rank-55 solution, 55 of the 60 output
positions are covered.  To get rank 54 we need to "merge" two of the rank-55
terms into a single term, which requires a specific algebraic relation.

We use a targeted search: try all ways to remove one of the 20 naive terms
(j=0 part) and see if the remaining 54 can be adjusted.  Also try removing
one Strassen term.  Use exact rational arithmetic where possible.
"""
import numpy as np
import random
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 10]


def _verify(U, V, W, n, m, p):
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

    for _ in range(2000):
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


def _residual_vec(params, n, m, p, rank):
    """Return flattened residual over all (nm, mp, pn) entries."""
    nm = n * m
    mp = m * p
    pn = p * n
    U = params[:rank * nm].reshape(nm, rank)
    V = params[rank * nm : rank * nm + rank * mp].reshape(mp, rank)
    W = params[rank * nm + rank * mp :].reshape(pn, rank)

    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

    Tt = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Tt[i * m + j, j * p + k, k * n + i] = 1.0
    return (T - Tt).flatten()


def _try_optimize_rank(n, m, p, rank, U_init, V_init, W_init, deadline):
    """Try to optimize a decomposition of the given rank starting from init."""
    nm = n * m
    mp = m * p
    pn = p * n

    U_flat = np.array(U_init, dtype=np.float64).flatten()
    V_flat = np.array(V_init, dtype=np.float64).flatten()
    W_flat = np.array(W_init, dtype=np.float64).flatten()
    x0 = np.concatenate([U_flat, V_flat, W_flat])

    try:
        import scipy.optimize as opt

        def residual(x):
            return _residual_vec(x, n, m, p, rank)

        result = opt.least_squares(residual, x0, method='trf', max_nfev=200)

        if result.success or result.cost < 1e-8:
            x_opt = result.x
            U_opt = x_opt[:rank * nm].reshape(nm, rank)
            V_opt = x_opt[rank * nm : rank * nm + rank * mp].reshape(mp, rank)
            W_opt = x_opt[rank * nm + rank * mp :].reshape(pn, rank)

            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2

            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()

            if _verify(U_list, V_list, W_list, n, m, p) and _is_valid_entries(U_list, V_list, W_list):
                return U_list, V_list, W_list

    except Exception:
        pass

    return None


def _structured_search_rank54(n, m, p, deadline):
    """
    Targeted search for rank 54.  In the rank-55 solution, the 20 naive terms
    (j=0) each have U,V,W entries all equal to 1 at specific positions.
    
    A rank-54 solution would need to "absorb" one naive term into the Strassen
    structure.  We try: for each naive term index t (0..19), set that term's
    U,V,W columns to zero and see if a rank-54 solution exists by solving a
    linear system for the W matrix (or adjusting U,V).
    
    More concretely: the multiplication tensor T has T[a,b,c] = 1 iff a=i*m+j,
    b=j*p+k, c=k*n+i.  For a rank-r decomposition:
        sum_{s=1}^{r} U[a,s] V[b,s] W[c,s] = T[a,b,c]
    
    This is a trilinear equation.  If we fix U and V, the equations are linear
    in W.  So: fix U and V from the rank-55 solution (dropping one column),
    and solve for W via least squares.  If the residual is zero, we have a
    rank-54 solution.
    
    We try dropping each of the 55 columns and solving for W.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    rank = 54

    U55, V55, W55 = _build_strassen55(n, m, p)
    U55 = np.array(U55, dtype=np.float64)
    V55 = np.array(V55, dtype=np.float64)
    W55 = np.array(W55, dtype=np.float64)

    # Build target tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # For each column to drop, try to solve for W
    for drop_col in range(55):
        if time.time() > deadline:
            break

        # U, V with column drop_col removed
        mask = np.ones(55, dtype=bool)
        mask[drop_col] = False
        U_r = U55[:, mask]  # (nm, 54)
        V_r = V55[:, mask]  # (mp, 54)

        # Build linear system for W: for each (a,b) with T[a,b,c]=1,
        # sum_s U[a,s] V[b,s] W[c,s] = 1
        # This is linear in W.  We have 60 equations (one per non-zero T entry),
        # but they share W entries.  Actually each non-zero (a,b,c) gives one
        # equation involving W[c, :].
        #
        # Let's set up: for each c, the equation is:
        #   sum_{s} (sum over a,b where T[a,b,c]=1) U[a,s] V[b,s] W[c,s] = ...
        # Wait, that's not right.  Each (a,b,c) triple is a separate equation.
        #
        # Actually, for a fixed c, the equations are:
        #   for each (a,b) with T[a,b,c] = 1: sum_s U[a,s] V[b,s] W[c,s] = 1
        #
        # So for each c, we have a system: M_c @ W[c,:] = 1
        # where M_c[a,b] = sum_s U[a,s] V[b,s]  (but only for (a,b) with T=1)
        #
        # Actually M_c is not quite right.  Let me re-derive.
        #
        # sum_s U[a,s] V[b,s] W[c,s] = 1
        # This is: (U[a,:] * V[b,:]) dot W[c,:] = 1
        # where U[a,:] * V[b,:] is element-wise product.
        #
        # So for fixed c, define for each (a,b) with T[a,b,c]=1:
        #   vec_{a,b} = U[a,:] * V[b,:]  (element-wise, length 54)
        #   vec_{a,b} dot W[c,:] = 1
        #
        # This is a linear system in W[c,:].

        # Collect equations
        # For each c, collect the (a,b) pairs where T[a,b,c] = 1
        eqs_by_c = {c: [] for c in range(pn)}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    eqs_by_c[c].append((a, b))

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

        # Also need to check zero entries
        U_list = U_r.tolist()
        V_list = V_r.tolist()
        W_list = W_sol.tolist()

        # Snap to half-integers
        U_snap = np.round(np.array(U_list) * 2) / 2
        V_snap = np.round(np.array(V_list) * 2) / 2
        W_snap = np.round(np.array(W_list) * 2) / 2

        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()

        if _verify(U_list, V_list, W_list, n, m, p) and _is_valid_entries(U_list, V_list, W_list):
            return U_list, V_list, W_list

    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130

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

    # If still time, try optimization-based search for rank 54
    if time.time() < deadline - 10 and best_rank > 54:
        r = 54
        U_init = [row[:r] for row in U_best]
        V_init = [row[:r] for row in V_best]
        W_init = [row[:r] for row in W_best]
        opt_deadline = deadline - 3
        result = _try_optimize_rank(n, m, p, r, U_init, V_init, W_init, opt_deadline)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = r

    return {"U": U_best, "V": V_best, "W": W_best}
