"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

Structural approach:
1. Build the rank-55 Strassen+schoolbook decomposition (known correct).
2. Form the structure tensor matrix M of shape (n*m*p, 55) where
   M[(i,j,k), c] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
3. The tensor equation is M @ x = 1 (all-ones vector of length 60).
4. Search for a 54-column submatrix M[:, S] of rank 54 such that
   M[:, S] @ x_S = 1 has a solution (i.e., 1 is in the column space).
   If found, the 54 terms with coefficients x_S give a rank-54 decomposition.
5. If no 54-subset works, return the rank-55 solution.

This is a structural linear-algebra search over the combinatorics of the
known decomposition, not an entry-wise optimization.
"""

import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 3, 10]


def _build_rank55(n, m, p):
    """Build the known rank-55 Strassen+schoolbook decomposition."""
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Strassen for 2x2 * 2x10 (5 blocks of 2 columns each)
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1

        u00, u01, u10, u11 = 0, 1, 3, 4
        v00, v01 = k0, k1
        v10, v11 = 10 + k0, 10 + k1
        w00, w01 = k0 * 2, k1 * 2
        w10, w11 = k0 * 2 + 1, k1 * 2 + 1

        # M1 = (A00+A11)(B00+B11)
        U[u00][col] = 1; U[u11][col] = 1
        V[v00][col] = 1; V[v11][col] = 1
        W[w00][col] = 1; W[w11][col] = 1
        col += 1

        # M2 = (A10+A11)B00
        U[u10][col] = 1; U[u11][col] = 1
        V[v00][col] = 1
        W[w10][col] = 1; W[w11][col] = -1
        col += 1

        # M3 = A00(B01-B11)
        U[u00][col] = 1
        V[v01][col] = 1; V[v11][col] = -1
        W[w01][col] = 1; W[w11][col] = 1
        col += 1

        # M4 = A11(B10-B00)
        U[u11][col] = 1
        V[v10][col] = 1; V[v00][col] = -1
        W[w00][col] = 1; W[w10][col] = 1
        col += 1

        # M5 = (A00+A01)B11
        U[u00][col] = 1; U[u01][col] = 1
        V[v11][col] = 1
        W[w01][col] = 1; W[w00][col] = -1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[u10][col] = 1; U[u00][col] = -1
        V[v00][col] = 1; V[v01][col] = 1
        W[w11][col] = 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[u01][col] = 1; U[u11][col] = -1
        V[v10][col] = 1; V[v11][col] = 1
        W[w00][col] = 1
        col += 1

    # Schoolbook for A2 (2x1) * B2 (1x10)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    assert col == rank
    return U, V, W


def _build_structure_matrix(U, V, W, n, m, p):
    """Build M of shape (n*m*p, rank) where M[(i,j,k),c] = U[i,j,c]*V[j,k,c]*W[k,i,c]."""
    rank = len(U[0])
    M = np.zeros((n * m * p, rank), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i * m + j) * p + k  # flatten (i,j,k)
                # Actually, the tensor index is (i,j,k) but we need consistent flattening
                # Let's use idx = i * (m*p) + j * p + k
                idx = i * (m * p) + j * p + k
                for c in range(rank):
                    M[idx, c] = U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
    return M


def _try_rank_reduction(M, target_rank, time_budget, seed=42):
    """
    Search for a target_rank-column submatrix of M (shape (60, 55)) such that
    the system M[:, S] @ x = ones(60) has a solution.
    Returns (subset_indices, coefficients) or (None, None).
    """
    n_eq = M.shape[0]
    n_cols = M.shape[1]
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    ones = np.ones(n_eq)

    # First, check: is the full system consistent? It should be (rank-55 is valid).
    # The all-ones coefficient vector should work: M @ ones = ones.
    # Verify: M @ np.ones(n_cols) should equal ones.

    # Strategy: try random subsets of size target_rank, check if rank == target_rank
    # and if ones is in the column space (i.e., residual of lstsq is ~0).
    # Number of C(55,54) = 55 subsets. We can check all of them!

    # Check all 54-subsets (removing one column at a time)
    for removed in range(n_cols):
        if time.time() > deadline - 5:
            break
        S = [c for c in range(n_cols) if c != removed]
        Ms = M[:, S]
        r = np.linalg.matrix_rank(Ms, tol=1e-10)
        if r < target_rank:
            continue
        # Check if ones is in column space: solve Ms @ x = ones, check residual
        x, residuals, rank_sol, sv = np.linalg.lstsq(Ms, ones, rcond=1e-10)
        residual = np.linalg.norm(Ms @ x - ones)
        if residual < 1e-8:
            return S, x

    # If no single-removal works, try random 54-subsets (but C(55,54)=55, so we've done all)
    # Actually for target_rank=54, we've checked all. For other ranks, use random.
    if target_rank < n_cols - 1:
        for attempt in range(5000):
            if time.time() > deadline - 5:
                break
            S = rng.choice(n_cols, size=target_rank, replace=False)
            Ms = M[:, S]
            r = np.linalg.matrix_rank(Ms, tol=1e-10)
            if r < target_rank:
                continue
            x, residuals, rank_sol, sv = np.linalg.lstsq(Ms, ones, rcond=1e-10)
            residual = np.linalg.norm(Ms @ x - ones)
            if residual < 1e-8:
                return S, x

    return None, None


def _reconstruct(U, V, W, S, coeffs, n, m, p):
    """Rebuild U,V,W with only the selected columns and scaled by coefficients."""
    target_rank = len(S)
    U_new = [[0] * target_rank for _ in range(n * m)]
    V_new = [[0] * target_rank for _ in range(m * p)]
    W_new = [[0] * target_rank for _ in range(p * n)]

    for new_c, old_c in enumerate(S):
        scale = coeffs[new_c]
        for row in range(n * m):
            U_new[row][new_c] = U[row][old_c] * scale
        for row in range(m * p):
            V_new[row][new_c] = V[row][old_c] * scale
        for row in range(p * n):
            W_new[row][new_c] = W[row][old_c] * scale

    return U_new, V_new, W_new


def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify decomposition using NumPy."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(Ua[i * m + j] * Va[j * p + k] * Wa[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = 145.0  # leave 5s margin before 150s kill

    n, m, p = DIMENSIONS

    # Step 1: Build rank-55 decomposition
    U, V, W = _build_rank55(n, m, p)

    # Verify rank-55 is correct
    if not _verify(U, V, W, n, m, p, tol=1e-8):
        # Should never happen, but fall back to schoolbook
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

    # Step 2: Build structure matrix
    M = _build_structure_matrix(U, V, W, n, m, p)

    # Step 3: Try to reduce to rank 54
    time_remaining = deadline - time.time()
    if time_remaining > 10:
        S, coeffs = _try_rank_reduction(M, 54, time_remaining - 5, seed=42)
        if S is not None:
            U54, V54, W54 = _reconstruct(U, V, W, S, coeffs, n, m, p)
            if _verify(U54, V54, W54, n, m, p, tol=1e-8):
                return {"U": U54, "V": V54, "W": W54}

    # Step 4: Try rank 53 (if time permits)
    time_remaining = deadline - time.time()
    if time_remaining > 10:
        S, coeffs = _try_rank_reduction(M, 53, time_remaining - 5, seed=123)
        if S is not None:
            U53, V53, W53 = _reconstruct(U, V, W, S, coeffs, n, m, p)
            if _verify(U53, V53, W53, n, m, p, tol=1e-8):
                return {"U": U53, "V": V53, "W": W53}

    # Step 5: Return rank-55 solution
    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
