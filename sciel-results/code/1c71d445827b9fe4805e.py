"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Strategy:
1. Constructive rank-18 baseline (2 Strassen blocks + schoolbook column).
2. Systematic rank-17 search using structured bases (Walsh-Hadamard on Z_2^2
   for the U factor) plus random ALS with rational polish.
3. If rank-17 fails, return the verified rank-18 solution.

The U factor is 4 x R. We fix U to a structured basis derived from the
Walsh-Hadamard transform of Z_2^2 (which diagonalizes the 2x2 matrix
multiplication structure), then solve linear systems for V and W.
"""

import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 5]


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    return T


def _check_decomposition(U, V, W, T, tol=1e-8):
    """Verify decomposition matches target tensor."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    R = U_arr.shape[1]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                total = 0.0
                for r in range(R):
                    total += U_arr[u, r] * V_arr[v, r] * W_arr[w, r]
                if abs(total - T[u, v, w]) > tol:
                    return False
    return True


def _check_exact(U, V, W, T):
    """Exact integer/half-integer verification using Fraction."""
    def to_frac(x):
        hx = round(x * 2)
        if abs(x - hx / 2.0) < 1e-7:
            return Fraction(int(hx), 2)
        f = Fraction(x).limit_denominator(64)
        if abs(x - float(f)) < 1e-7:
            return f
        return Fraction(x).limit_denominator(64)

    Uf = [[to_frac(x) for x in row] for row in U]
    Vf = [[to_frac(x) for x in row] for row in V]
    Wf = [[to_frac(x) for x in row] for row in W]
    R = len(U[0])
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                total = Fraction(0)
                for r in range(R):
                    total += Uf[u][r] * Vf[v][r] * Wf[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _build_strassen_block(col0, col1, base, U, V, W, n, m, p):
    """Add 7 Strassen terms for 2x2 x 2x2 block with B columns col0, col1."""
    r = base
    U[0][r] += 1; U[3][r] += 1
    V[0*p+col0][r] += 1; V[1*p+col1][r] += 1
    W[col0*n+0][r] += 1; W[col1*n+1][r] += 1

    r = base + 1
    U[2][r] += 1; U[3][r] += 1
    V[0*p+col0][r] += 1
    W[col0*n+1][r] += 1; W[col1*n+1][r] += -1

    r = base + 2
    U[0][r] += 1
    V[0*p+col1][r] += 1; V[1*p+col1][r] += -1
    W[col1*n+0][r] += 1; W[col1*n+1][r] += 1

    r = base + 3
    U[3][r] += 1
    V[1*p+col0][r] += 1; V[0*p+col0][r] += -1
    W[col0*n+0][r] += 1; W[col0*n+1][r] += 1

    r = base + 4
    U[0][r] += 1; U[1][r] += 1
    V[1*p+col1][r] += 1
    W[col0*n+0][r] += -1; W[col1*n+0][r] += 1

    r = base + 5
    U[2][r] += 1; U[0][r] += -1
    V[0*p+col0][r] += 1; V[0*p+col1][r] += 1
    W[col1*n+1][r] += 1

    r = base + 6
    U[1][r] += 1; U[3][r] += -1
    V[1*p+col0][r] += 1; V[1*p+col1][r] += 1
    W[col0*n+0][r] += 1


def _build_rank18():
    """Construct the rank-18 decomposition: 2 Strassen blocks + schoolbook for col 4."""
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    _build_strassen_block(0, 1, 0, U, V, W, n, m, p)
    _build_strassen_block(2, 3, 7, U, V, W, n, m, p)

    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1

    return U, V, W


def _walsh_hadamard_basis(R):
    """
    Generate structured U bases using Walsh-Hadamard patterns on Z_2^2.
    The 4 rows of U correspond to A[0,0], A[0,1], A[1,0], A[1,1].
    Walsh-Hadamard characters of Z_2^2 give 4 orthogonal sign patterns.
    We use these as column patterns, allowing linear combinations.
    """
    # Walsh-Hadamard matrix of order 4 (Hadamard on Z_2^2)
    # H = [[1, 1, 1, 1],
    #      [1, -1, 1, -1],
    #      [1, 1, -1, -1],
    #      [1, -1, -1, 1]]
    H = np.array([
        [1,  1,  1,  1],
        [1, -1,  1, -1],
        [1,  1, -1, -1],
        [1, -1, -1,  1]
    ], dtype=np.float64)

    bases = []

    # Basis 1: Pure Walsh-Hadamard columns (first R columns of H repeated/extended)
    if R <= 4:
        bases.append(H[:, :R].tolist())
    elif R <= 8:
        bases.append(np.hstack([H, H]).tolist()[:4])  # 4 x 8
        # Actually need 4 x R
        bases[-1] = np.hstack([H, H[:, :R-4]]).tolist()
    else:
        # Tile H
        n_tiles = (R + 3) // 4
        U_wh = np.tile(H, (1, n_tiles))[:, :R]
        bases.append(U_wh.tolist())

    # Basis 2: Identity + random perturbations
    np.random.seed(12345)
    for seed in range(10):
        np.random.seed(seed)
        U_rand = np.random.randn(4, R) * 0.5
        # Ensure some structure: make first 4 columns close to identity
        U_rand[:, :min(4, R)] += np.eye(4, min(4, R))
        bases.append(U_rand.tolist())

    # Basis 3: Strassen-like patterns
    # Strassen uses specific linear combinations of A entries
    # M1: A00+A11, M2: A10+A11, M3: A00, M4: A11, M5: A00+A01, M6: A10-A00, M7: A01-A11
    strassen_patterns = [
        [1, 0, 0, 1],  # A00+A11
        [0, 0, 1, 1],  # A10+A11
        [1, 0, 0, 0],  # A00
        [0, 0, 0, 1],  # A11
        [1, 1, 0, 0],  # A00+A01
        [0, 0, 1, -1], # A10-A00
        [0, 1, 0, -1], # A01-A11
    ]
    # Build a 4 x R matrix using these patterns cyclically
    U_strassen = np.zeros((4, R))
    for r in range(R):
        pat = strassen_patterns[r % 7]
        U_strassen[:, r] = pat
    bases.append(U_strassen.tolist())

    return bases


def _solve_VW_given_U(U, T, R):
    """
    Given fixed U (4 x R), solve for V (10 x R) and W (10 x R).
    
    The constraint is: for each (u,v,w) with T[u,v,w]=1:
        sum_r U[u,r] * V[v,r] * W[w,r] = 1
    For T[u,v,w]=0:
        sum_r U[u,r] * V[v,r] * W[w,r] = 0
    
    This is bilinear in V and W. We can fix V and solve for W linearly,
    or use ALS.
    """
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n * m, m * p, p * n
    U_arr = np.array(U, dtype=np.float64)

    # ALS: alternate solving for V and W given the other
    np.random.seed(42)
    V = np.random.randn(V_rows, R) * 0.1
    W = np.random.randn(W_rows, R) * 0.1

    T_float = T.astype(np.float64)

    for it in range(200):
        # Solve for V: for each v, sum_u,w T[u,v,w] * W[w,r] * U[u,r] * V[v,r] ... 
        # Actually: T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
        # Fix U, W: T[:,v,:] = sum_r (U[:,r] * W[:,r]) * V[v,r]
        # So for each v: T[:,v,:].reshape(-1) = V[v,:] @ (U*W).reshape(R, -1)
        # i.e., A @ V[v,:] = b where A[r, (u,w)] = U[u,r]*W[w,r]
        A_v = np.einsum('ur,wr->urw', U_arr, W).reshape(R, -1)
        for v in range(V_rows):
            b = T_float[:, v, :].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(A_v, b, rcond=None)
            V[v] = sol

        # Solve for W: for each w, T[:,:,w].reshape(-1) = W[w,:] @ (U*V).reshape(R, -1)
        A_w = np.einsum('ur,vr->uvr', U_arr, V).reshape(R, -1)
        for w in range(W_rows):
            b = T_float[:, :, w].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(A_w, b, rcond=None)
            W[w] = sol

        # Check convergence
        result = np.einsum('ur,vr,wr->uvw', U_arr, V, W)
        err = np.linalg.norm(result - T_float)
        if err < 1e-10:
            break

    result = np.einsum('ur,vr,wr->uvw', U_arr, V, W)
    err = np.linalg.norm(result - T_float)
    if err < 1e-7:
        return V.tolist(), W.tolist()
    return None


def _try_rank17_structured(deadline):
    """Try rank-17 using structured U bases and solving for V, W."""
    n, m, p = DIMENSIONS
    T = _build_target()
    R = 17

    bases = _walsh_hadamard_basis(R)

    for U_list in bases:
        if time.time() > deadline:
            break
        U_arr = np.array(U_list, dtype=np.float64)
        if U_arr.shape != (4, R):
            continue

        result = _solve_VW_given_U(U_arr, T, R)
        if result is not None:
            V_list, W_list = result
            # Verify
            if _check_decomposition(U_list, V_list, W_list, T, tol=1e-7):
                # Try to snap to rationals
                for denom in [1, 2, 4]:
                    U_s = np.round(U_arr * denom) / denom
                    V_s = np.round(np.array(V_list) * denom) / denom
                    W_s = np.round(np.array(W_list) * denom) / denom
                    if _check_exact(U_s.tolist(), V_s.tolist(), W_s.tolist(), T):
                        return U_s.tolist(), V_s.tolist(), W_s.tolist()
                return U_list, V_list, W_list

    return None


def _try_rank17_als(deadline, n_seeds=30, max_iters=100):
    """Try ALS for rank 17 with multiple seeds."""
    n, m, p = DIMENSIONS
    T = _build_target()
    R = 17
    U_rows, V_rows, W_rows = n * m, m * p, p * n
    T_float = T.astype(np.float64)

    for seed in range(n_seeds):
        if time.time() > deadline:
            break
        np.random.seed(seed)

        U = np.random.randn(U_rows, R) * 0.3
        V = np.random.randn(V_rows, R) * 0.3
        W = np.random.randn(W_rows, R) * 0.3

        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Update U
            A_u = np.einsum('vr,wr->vwr', V, W).reshape(-1, R)
            for u in range(U_rows):
                b = T_float[u].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A_u, b, rcond=None)
                U[u] = sol

            # Update V
            A_v = np.einsum('ur,wr->uwr', U, W).reshape(-1, R)
            for v in range(V_rows):
                b = T_float[:, v, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A_v, b, rcond=None)
                V[v] = sol

            # Update W
            A_w = np.einsum('ur,vr->uvr', U, V).reshape(-1, R)
            for w in range(W_rows):
                b = T_float[:, :, w].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A_w, b, rcond=None)
                W[w] = sol

            result = np.einsum('ur,vr,wr->uvw', U, V, W)
            err = np.linalg.norm(result - T_float)

            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err

        if _check_decomposition(U.tolist(), V.tolist(), W.tolist(), T, tol=1e-7):
            # Try to snap to rationals
            for denom in [1, 2]:
                U_s = np.round(U * denom) / denom
                V_s = np.round(V * denom) / denom
                W_s = np.round(W * denom) / denom
                if _check_exact(U_s.tolist(), V_s.tolist(), W_s.tolist(), T):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
            return U.tolist(), V.tolist(), W.tolist()

    return None


def _try_rank16_als(deadline, n_seeds=15, max_iters=80):
    """Try ALS for rank 16."""
    n, m, p = DIMENSIONS
    T = _build_target()
    R = 16
    U_rows, V_rows, W_rows = n * m, m * p, p * n
    T_float = T.astype(np.float64)

    for seed in range(n_seeds):
        if time.time() > deadline:
            break
        np.random.seed(1000 + seed)

        U = np.random.randn(U_rows, R) * 0.3
        V = np.random.randn(V_rows, R) * 0.3
        W = np.random.randn(W_rows, R) * 0.3

        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline:
                break

            A_u = np.einsum('vr,wr->vwr', V, W).reshape(-1, R)
            for u in range(U_rows):
                b = T_float[u].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A_u, b, rcond=None)
                U[u] = sol

            A_v = np.einsum('ur,wr->uwr', U, W).reshape(-1, R)
            for v in range(V_rows):
                b = T_float[:, v, :].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A_v, b, rcond=None)
                V[v] = sol

            A_w = np.einsum('ur,vr->uvr', U, V).reshape(-1, R)
            for w in range(W_rows):
                b = T_float[:, :, w].reshape(-1)
                sol, _, _, _ = np.linalg.lstsq(A_w, b, rcond=None)
                W[w] = sol

            result = np.einsum('ur,vr,wr->uvw', U, V, W)
            err = np.linalg.norm(result - T_float)

            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err

        if _check_decomposition(U.tolist(), V.tolist(), W.tolist(), T, tol=1e-7):
            for denom in [1, 2]:
                U_s = np.round(U * denom) / denom
                V_s = np.round(V * denom) / denom
                W_s = np.round(W * denom) / denom
                if _check_exact(U_s.tolist(), V_s.tolist(), W_s.tolist(), T):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
            return U.tolist(), V.tolist(), W.tolist()

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    T = _build_target()

    # First, get the rank-18 baseline
    U18, V18, W18 = _build_rank18()
    if not _check_decomposition(U18, V18, W18, T, tol=1e-8):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rank-17 with structured bases
    if time.time() < deadline - 30:
        result17 = _try_rank17_structured(deadline - 30)
        if result17 is not None:
            U17, V17, W17 = result17
            if _check_decomposition(U17, V17, W17, T, tol=1e-8):
                return {"U": U17, "V": V17, "W": W17}

    # Try rank-17 with ALS
    if time.time() < deadline - 20:
        result17_als = _try_rank17_als(deadline - 20, n_seeds=25, max_iters=80)
        if result17_als is not None:
            U17, V17, W17 = result17_als
            if _check_decomposition(U17, V17, W17, T, tol=1e-8):
                return {"U": U17, "V": V17, "W": W17}

    # Try rank-16 with ALS
    if time.time() < deadline - 10:
        result16 = _try_rank16_als(deadline - 10, n_seeds=10, max_iters=60)
        if result16 is not None:
            U16, V16, W16 = result16
            if _check_decomposition(U16, V16, W16, T, tol=1e-8):
                return {"U": U16, "V": V16, "W": W16}

    # Return rank-18 solution
    return {"U": U18, "V": V18, "W": W18}
