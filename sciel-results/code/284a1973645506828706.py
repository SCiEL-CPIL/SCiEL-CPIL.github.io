"""
Rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.

Combined approach:
  1. Sparse ALS exploiting the 63 non-zero entries structure.
  2. Half-integer rounding with verification.
  3. High-precision polish pass for near-converged solutions.
  4. Multiple restarts with diverse initializations.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 7]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin

    # Precompute target tensor indices
    nz_a, nz_b, nz_c = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz_a.append(i * m + j)
                nz_b.append(j * p + k)
                nz_c.append(k * n + i)
    nz_a = np.array(nz_a, dtype=int)
    nz_b = np.array(nz_b, dtype=int)
    nz_c = np.array(nz_c, dtype=int)
    nz_vals = np.ones(len(nz_a))

    # Group indices by each dimension for efficient ALS
    # For each a, which (b,c) pairs are non-zero?
    groups_a = {}  # a -> list of (b, c, val)
    groups_b = {}  # b -> list of (a, c, val)
    groups_c = {}  # c -> list of (a, b, val)
    for idx in range(len(nz_a)):
        a, b, c = nz_a[idx], nz_b[idx], nz_c[idx]
        groups_a.setdefault(a, []).append((b, c, 1.0))
        groups_b.setdefault(b, []).append((a, c, 1.0))
        groups_c.setdefault(c, []).append((a, b, 1.0))

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    def is_valid_decomposition(U, V, W, tol=1e-9):
        """Check exactness efficiently."""
        T = np.einsum('ar,br,cr->abc', U, V, W)
        for idx in range(len(nz_a)):
            a, b, c = nz_a[idx], nz_b[idx], nz_c[idx]
            if abs(T[a, b, c] - 1.0) > tol:
                return False
        # Check off-diagonal
        mask = np.zeros((n * m, m * p, p * n), dtype=bool)
        mask[nz_a, nz_b, nz_c] = True
        if np.any(np.abs(T[~mask]) > tol):
            return False
        return True

    def round_to_half(U, V, W):
        return np.round(U * 2) / 2, np.round(V * 2) / 2, np.round(W * 2) / 2

    # Trivial rank-63 decomposition
    trivial_rank = n * m * p
    U_t = np.zeros((n * m, trivial_rank))
    V_t = np.zeros((m * p, trivial_rank))
    W_t = np.zeros((p * n, trivial_rank))
    for idx in range(trivial_rank):
        U_t[nz_a[idx], idx] = 1.0
        V_t[nz_b[idx], idx] = 1.0
        W_t[nz_c[idx], idx] = 1.0

    def als_sparse(R, max_iters=50, tol=1e-12, seed=0):
        """
        Efficient ALS exploiting sparsity.
        """
        rng = np.random.RandomState(seed)
        # Initialize with values near half-integers for better snapping
        U = rng.randint(0, 3, size=(n * m, R)) * 0.5 - 0.5
        V = rng.randint(0, 3, size=(m * p, R)) * 0.5 - 0.5
        W = rng.randint(0, 3, size=(p * n, R)) * 0.5 - 0.5

        for it in range(max_iters):
            # --- Solve for U ---
            # For each a: find U[a,:] such that sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # Build M_VW: for each r, the vector over (b,c) of V[b,r]*W[c,r]
            M_VW = np.einsum('br,cr->rbc', V, W).reshape(R, m * p * p * n)
            for a in range(n * m):
                rhs = np.zeros(m * p * p * n)
                for (b, c, val) in groups_a.get(a, []):
                    rhs[b * (p * n) + c] = val
                try:
                    U[a, :] = np.linalg.lstsq(M_VW.T, rhs, rcond=None)[0]
                except:
                    pass

            # --- Solve for V ---
            M_UW = np.einsum('ar,cr->rac', U, W).reshape(R, n * m * p * n)
            for b in range(m * p):
                rhs = np.zeros(n * m * p * n)
                for (a, c, val) in groups_b.get(b, []):
                    rhs[a * (p * n) + c] = val
                try:
                    V[b, :] = np.linalg.lstsq(M_UW.T, rhs, rcond=None)[0]
                except:
                    pass

            # --- Solve for W ---
            M_UV = np.einsum('ar,br->rab', U, V).reshape(R, n * m * m * p)
            for c in range(p * n):
                rhs = np.zeros(n * m * m * p)
                for (a, b, val) in groups_c.get(c, []):
                    rhs[a * (m * p) + b] = val
                try:
                    W[c, :] = np.linalg.lstsq(M_UV.T, rhs, rcond=None)[0]
                except:
                    pass

            # Check convergence
            T_cur = np.einsum('ar,br,cr->abc', U, V, W)
            err = 0.0
            for idx in range(len(nz_a)):
                a, b, c = nz_a[idx], nz_b[idx], nz_c[idx]
                err += (T_cur[a, b, c] - 1.0) ** 2
            mask = np.zeros((n * m, m * p, p * n), dtype=bool)
            mask[nz_a, nz_b, nz_c] = True
            err += np.sum(T_cur[~mask] ** 2)
            if err < tol:
                break

        return U, V, W

    def polish(U, V, W, R, max_iters=20, tol=1e-14):
        """High-precision polish pass."""
        for it in range(max_iters):
            M_VW = np.einsum('br,cr->rbc', V, W).reshape(R, m * p * p * n)
            for a in range(n * m):
                rhs = np.zeros(m * p * p * n)
                for (b, c, val) in groups_a.get(a, []):
                    rhs[b * (p * n) + c] = val
                try:
                    U[a, :] = np.linalg.lstsq(M_VW.T, rhs, rcond=None)[0]
                except:
                    pass

            M_UW = np.einsum('ar,cr->rac', U, W).reshape(R, n * m * p * n)
            for b in range(m * p):
                rhs = np.zeros(n * m * p * n)
                for (a, c, val) in groups_b.get(b, []):
                    rhs[a * (p * n) + c] = val
                try:
                    V[b, :] = np.linalg.lstsq(M_UW.T, rhs, rcond=None)[0]
                except:
                    pass

            M_UV = np.einsum('ar,br->rab', U, V).reshape(R, n * m * m * p)
            for c in range(p * n):
                rhs = np.zeros(n * m * m * p)
                for (a, b, val) in groups_c.get(c, []):
                    rhs[a * (m * p) + b] = val
                try:
                    W[c, :] = np.linalg.lstsq(M_UV.T, rhs, rcond=None)[0]
                except:
                    pass

            T_cur = np.einsum('ar,br,cr->abc', U, V, W)
            err = 0.0
            for idx in range(len(nz_a)):
                a, b, c = nz_a[idx], nz_b[idx], nz_c[idx]
                err += (T_cur[a, b, c] - 1.0) ** 2
            mask = np.zeros((n * m, m * p, p * n), dtype=bool)
            mask[nz_a, nz_b, nz_c] = True
            err += np.sum(T_cur[~mask] ** 2)
            if err < tol:
                break

        return U, V, W

    best_U, best_V, best_W = U_t, V_t, W_t
    best_rank = trivial_rank

    # Try ranks from 30 down to 18
    for R in range(30, 17, -1):
        if time.time() > deadline - 25:
            break
        for trial in range(8):
            if time.time() > deadline - 10:
                break
            seed = trial * 104729 + R * 7919
            U, V, W = als_sparse(R, max_iters=40, tol=1e-13, seed=seed)
            
            # Polish pass
            U, V, W = polish(U, V, W, R, max_iters=15, tol=1e-14)
            
            # Round and verify
            U_r, V_r, W_r = round_to_half(U, V, W)
            if is_valid_decomposition(U_r, V_r, W_r, tol=1e-8):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = R
                break  # try lower rank

    if best_rank < trivial_rank:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    return {"U": to_list(U_t), "V": to_list(V_t), "W": to_list(W_t)}
