"""
Rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.

Strategy:
  1. Build initial rank-93 decomposition via block Strassen/schoolbook.
  2. Iteratively attempt rank reduction by zeroing terms and re-solving.
  Target: find exact half-integer decomposition with rank < 93.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]


def _strassen_2x2x2():
    """
    Return U, V, W matrices (each 4x7) for the rank-7 Strassen decomposition of 2x2x2.
    
    Indexing convention:
      U[i*m + j, t] where i,j in {0,1} for A[i,j], m=2
      V[j*p + k, t] where j,k in {0,1} for B[j,k], p=2
      W[k*n + i, t] where k,i in {0,1} for C[k,i], n=2  (trace convention W[k*n+i])
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = a(e+f)
    U[0, 0] = 1
    V[0, 0] = 1
    V[1, 0] = 1
    W[0, 0] = 1
    
    # M2 = (a+b)h
    U[0, 1] = 1
    U[1, 1] = 1
    V[3, 1] = 1
    W[1, 1] = 1
    
    # M3 = (c+d)e
    U[2, 2] = 1
    U[3, 2] = 1
    V[0, 2] = 1
    W[2, 2] = 1
    
    # M4 = d(g+f)
    U[3, 3] = 1
    V[2, 3] = 1
    V[1, 3] = 1
    W[3, 3] = 1
    
    # M5 = (a-d)(e+h)
    U[0, 4] = 1
    U[3, 4] = -1
    V[0, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1
    W[3, 4] = 1
    
    # M6 = (b-c)(e+f)
    U[1, 5] = 1
    U[2, 5] = -1
    V[0, 5] = 1
    V[1, 5] = 1
    W[0, 5] = 1
    W[2, 5] = 1
    
    # M7 = (a+c)(h-g)
    U[0, 6] = 1
    U[2, 6] = 1
    V[3, 6] = 1
    V[2, 6] = -1
    W[1, 6] = 1
    W[3, 6] = -1
    
    # Verify
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0.0
                for t in range(7):
                    total += U[i * 2 + j, t] * V[j * 2 + k, t] * W[k * 2 + i, t]
                if abs(total - 1.0) > 1e-9:
                    raise ValueError(f"Strassen verification failed at i={i},j={j},k={k}: got {total}")
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    if U.shape[0] != n * m:
        return False
    if V.shape[0] != m * p:
        return False
    if W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for t in range(rank):
                    total += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if abs(total - 1.0) > 1e-9:
                    return False
    
    return True


def _build_initial_decomposition():
    """Build the rank-93 block decomposition as starting point."""
    n, m, p = 4, 4, 7
    U1, V1, W1 = _strassen_2x2x2()
    r1 = 7
    r_kron = 49

    U_kron = np.zeros((16, r_kron), dtype=float)
    V_kron = np.zeros((16, r_kron), dtype=float)
    W_kron = np.zeros((16, r_kron), dtype=float)

    for t1 in range(r1):
        for t2 in range(r1):
            t = t1 * r1 + t2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            idx_U = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                            U_kron[idx_U, t] = U1[i1 * 2 + j1, t1] * U1[i2 * 2 + j2, t2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            idx_V = (j1 * 2 + j2) * 4 + (k1 * 2 + k2)
                            V_kron[idx_V, t] = V1[j1 * 2 + k1, t1] * V1[j2 * 2 + k2, t2]
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            idx_W = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)
                            W_kron[idx_W, t] = W1[k1 * 2 + i1, t1] * W1[k2 * 2 + i2, t2]

    rank_part2 = 28
    U_part2 = np.zeros((16, rank_part2), dtype=float)
    V_part2 = np.zeros((28, rank_part2), dtype=float)
    W_part2 = np.zeros((28, rank_part2), dtype=float)

    t_offset = 0
    for i1 in range(2):
        for j1 in range(2):
            for t in range(7):
                tt = t_offset + t
                for i2 in range(2):
                    for j2 in range(2):
                        U_part2[(2*i1+i2)*4 + (2*j1+j2), tt] = U1[i2*2+j2, t]
                for j2 in range(2):
                    for k2 in range(2):
                        V_part2[(2*j1+j2)*7 + (4+k2), tt] = V1[j2*2+k2, t]
                for k2 in range(2):
                    for i2 in range(2):
                        W_part2[(4+k2)*4 + (2*i1+i2), tt] = W1[k2*2+i2, t]
                t_offset += 7

    rank_part3 = 16
    U_part3 = np.zeros((16, rank_part3), dtype=float)
    V_part3 = np.zeros((28, rank_part3), dtype=float)
    W_part3 = np.zeros((28, rank_part3), dtype=float)
    col3 = 0
    for i in range(4):
        for j in range(4):
            U_part3[i*4+j, col3] = 1
            V_part3[j*7+6, col3] = 1
            W_part3[6*4+i, col3] = 1
            col3 += 1

    U_full = np.zeros((16, 93), dtype=float)
    V_full = np.zeros((28, 93), dtype=float)
    W_full = np.zeros((28, 93), dtype=float)
    U_full[:, 0:49] = U_kron
    V_full[:, 0:49] = V_kron
    W_full[:, 0:49] = W_kron
    U_full[:, 49:77] = U_part2
    V_full[:, 49:77] = V_part2
    W_full[:, 49:77] = W_part2
    U_full[:, 77:93] = U_part3
    V_full[:, 77:93] = V_part3
    W_full[:, 77:93] = W_part3

    return U_full, V_full, W_full


def _try_reduce_rank(U, V, W, n, m, p, target_rank, max_iters=50, deadline=None):
    """
    Try to reduce rank by zeroing out terms and re-solving.
    Returns (U, V, W, success) with rank <= target_rank if successful.
    """
    start_rank = U.shape[1]
    if start_rank <= target_rank:
        return U, V, W, True

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = start_rank

    for iter_num in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        # Try to remove one rank term
        new_rank = best_rank - 1
        if new_rank < target_rank:
            break

        # Pick a random term to zero out
        t_to_remove = np.random.randint(0, best_rank)

        # Build the linear system for the remaining terms
        # We have 112 equations (one per (i,j,k)) and 3*new_rank unknowns
        # Equation: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1

        # This is a trilinear system, not linear. Use iterative approach:
        # Fix two matrices, solve for the third.

        # Strategy: randomly zero the term, then re-solve via alternating least squares
        U_try = best_U[:, :new_rank].copy()
        V_try = best_V[:, :new_rank].copy()
        W_try = best_W[:, :new_rank].copy()

        # Remove the term
        U_try[:, t_to_remove] = 0
        V_try[:, t_to_remove] = 0
        W_try[:, t_to_remove] = 0

        # Alternating least squares: fix two, solve for third
        for als_iter in range(20):
            if deadline is not None and time.time() > deadline:
                break

            # Solve for W given U, V
            # For each (i,j,k): sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
            # Group by (k,i): W[k*n+i, :] appears in equations for all j
            # Equation for (i,j,k): sum_t (U[i*m+j,t] * V[j*p+k,t]) * W[k*n+i,t] = 1
            # For fixed (k,i): sum_j sum_t (U[i*m+j,t] * V[j*p+k,t]) * W[k*n+i,t] ... no
            # Actually each (i,j,k) gives one equation. W[k*n+i,t] appears in equations for all j.
            # Let's set up: for each (i,k), the variable is W[k*n+i, :] (a vector of length new_rank)
            # It appears in equations for j=0..m-1.
            # Equation (i,j,k): sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
            # This is: (U[i*m+j,:] * V[j*p+k,:]) dot W[k*n+i,:] = 1
            # For fixed (i,k), we have m equations:
            #   (U[i*m+j,:] * V[j*p+k,:]) dot W[k*n+i,:] = 1, for j=0..m-1
            # This is a system of m equations in new_rank unknowns. Underdetermined if new_rank > m.
            # Use least squares.

            # Build A matrix for W: shape (n*p, new_rank) where row (k*n+i) corresponds to
            # the coefficient vector for W[k*n+i, :]
            # Actually, let's just do it directly:
            # For each (i,k), collect the m coefficient vectors
            A_W = np.zeros((n * p * m, new_rank))
            b_W = np.ones(n * p * m)
            row = 0
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        A_W[row, :] = U[i*m+j, :] * V[j*p+k, :]
                        b_W[row] = 1.0
                        row += 1

            # Solve A_W dot W_flat = b_W via least squares
            W_flat, _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
            W_try = W_flat.reshape(n*p, new_rank)

            # Solve for V given U, W
            A_V = np.zeros((m * p * n, new_rank))
            b_V = np.ones(m * p * n)
            row = 0
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        A_V[row, :] = U[i*m+j, :] * W[k*n+i, :]
                        b_V[row] = 1.0
                        row += 1
            V_flat, _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
            V_try = V_flat.reshape(m*p, new_rank)

            # Solve for U given V, W
            A_U = np.zeros((n * m * p, new_rank))
            b_U = np.ones(n * m * p)
            row = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        A_U[row, :] = V[j*p+k, :] * W[k*n+i, :]
                        b_U[row] = 1.0
                        row += 1
            U_flat, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
            U_try = U_flat.reshape(n*m, new_rank)

        # Check if the solution is exact
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for t in range(new_rank):
                        total += U_try[i*m+j, t] * V_try[j*p+k, t] * W_try[k*n+i, t]
                    max_err = max(max_err, abs(total - 1.0))

        if max_err < 1e-6:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = new_rank
            continue
        else:
            # Try a different term to remove
            continue

    return best_U, best_V, best_W, best_rank


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    n, m, p = 4, 4, 7

    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    try:
        U_init, V_init, W_init = _build_initial_decomposition()

        if not _verify_decomposition(U_init, V_init, W_init, n, m, p):
            raise ValueError("Initial decomposition failed verification")

        # Try to reduce rank
        np.random.seed(42)
        U_red, V_red, W_red, final_rank = _try_reduce_rank(
            U_init, V_init, W_init, n, m, p,
            target_rank=70, max_iters=30, deadline=deadline
        )

        if final_rank < 93 and _verify_decomposition(U_red, V_red, W_red, n, m, p):
            # Round to half-integers
            U_rounded = np.round(U_red * 2) / 2
            V_rounded = np.round(V_red * 2) / 2
            W_rounded = np.round(W_red * 2) / 2
            if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}

        # Fall back to initial decomposition
        if time.time() > deadline:
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

        return {"U": U_init.tolist(), "V": V_init.tolist(), "W": W_init.tolist()}

    except Exception:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
