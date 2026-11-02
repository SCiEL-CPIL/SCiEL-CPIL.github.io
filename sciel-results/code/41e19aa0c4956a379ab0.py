"""
Rank-14 exact tensor decomposition for 2x2 * 2x4 matrix multiplication.
We keep the known Strassen-based rank-14 solution and attempt a short numerical
search for a rank-13 decomposition.  If the search succeeds (and the found
decomposition can be rounded to integer/half-integer entries and verified exactly),
we return rank 13; otherwise we fall back to rank 14.
"""

import time
import random
import math
import numpy as np

DIMENSIONS = [2, 2, 4]

# ---------------------------------------------------------------------------
# Build the target multiplication tensor T of shape (4, 8, 8).
# Index convention:
#   a = 2*i + j      (i in {0,1}, j in {0,1})
#   b = 4*j + k      (j in {0,1}, k in {0,1,2,3})
#   c = 2*k + i      (k in {0,1,2,3}, i in {0,1})
# T[a, b, c] = 1 for each (i, j, k), else 0.
# ---------------------------------------------------------------------------
def build_target_tensor():
    n, m, p = 2, 2, 4
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = 2 * i + j
                b = 4 * j + k
                c = 2 * k + i
                T[a, b, c] = 1.0
    return T

# ---------------------------------------------------------------------------
# Verify that the factor matrices exactly decompose the target tensor.
# We check that sum_r U[:,r] V[:,r] W[:,r] == T within tight tolerance.
# ---------------------------------------------------------------------------
def verify_decomposition(U, V, W, T, tol=1e-9):
    # Reconstruct tensor
    rank = U.shape[1]
    T_recon = np.zeros(T.shape, dtype=np.float64)
    for r in range(rank):
        T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    err = np.max(np.abs(T_recon - T))
    return err < tol

# ---------------------------------------------------------------------------
# Try to round entries to the nearest integer or half-integer and check.
# ---------------------------------------------------------------------------
def round_to_half_int(X):
    # Round to nearest 0.5
    return np.round(2.0 * X) / 2.0

# ---------------------------------------------------------------------------
# Known rank-14 decomposition (Strassen on two 2x2x2 blocks).
# ---------------------------------------------------------------------------
def construct_rank14():
    n, m, p = 2, 2, 4
    rank = 14

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_term(u_vals, v_vals, w_vals, col):
        for i in range(n * m):
            U[i][col] = u_vals[i]
        for j in range(m * p):
            V[j][col] = v_vals[j]
        for k in range(p * n):
            W[k][col] = w_vals[k]

    # Left Part (Cols 0..6)
    add_term([1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0], 0)
    add_term([0, 0, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, -1, 0, 0, 0, 0], 1)
    add_term([1, 0, 0, 0], [0, 1, 0, 0, 0, -1, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0], 2)
    add_term([0, 0, 0, 1], [-1, 0, 0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], 3)
    add_term([1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [-1, 0, 1, 0, 0, 0, 0, 0], 4)
    add_term([-1, 0, 1, 0], [1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], 5)
    add_term([0, 1, 0, -1], [0, 0, 0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], 6)

    # Right Part (Cols 7..13)
    add_term([1, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 0, 0, 1], 7)
    add_term([0, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, -1], 8)
    add_term([1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 1, 1], 9)
    add_term([0, 0, 0, 1], [0, 0, -1, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1, 0, 0], 10)
    add_term([1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, -1, 0, 1, 0], 11)
    add_term([-1, 0, 1, 0], [0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], 12)
    add_term([0, 1, 0, -1], [0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 0, 0, 0], 13)

    return {"U": U, "V": V, "W": W}

# ---------------------------------------------------------------------------
# Attempt a numerical search for a rank-13 decomposition.
# We use a simple gradient-descent approach with multiple random restarts.
# Objective: minimize || sum_r U_r V_r W_r - T ||_F^2
# ---------------------------------------------------------------------------
def search_rank13(T, time_budget=60.0):
    n, m, p = 2, 2, 4
    rank = 13
    T_flat = T.ravel()
    N = T_flat.shape[0]  # 256

    best_err = float('inf')
    best_factors = None

    start_time = time.time()
    num_restarts = 50
    max_iter = 2000
    lr = 0.01
    momentum = 0.9

    for restart in range(num_restarts):
        if time.time() - start_time > time_budget:
            break

        random.seed(42 + restart)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5

        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)

        for it in range(max_iter):
            if time.time() - start_time > time_budget:
                break

            # Compute tensor reconstruction
            T_recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
            for r in range(rank):
                T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            T_recon_flat = T_recon.ravel()
            err = np.linalg.norm(T_recon_flat - T_flat)
            if err < best_err:
                best_err = err
                best_factors = (U.copy(), V.copy(), W.copy())

            if err < 1e-8:
                break

            # Compute gradients
            dT = T_recon_flat - T_flat
            dT_reshaped = dT.reshape(n*m, m*p, p*n)

            gU = np.zeros_like(U)
            gV = np.zeros_like(V)
            gW = np.zeros_like(W)

            for r in range(rank):
                # d/dU[:,r] = sum over j,k: dT[i,j,k] V[j,r] W[k,r]
                gU[:, r] = np.einsum('ijk,jr,kr->ir', dT_reshaped, V[:, r], W[:, r]).reshape(n*m, rank)[:, r]
                # Actually, let's do it properly
                gU[:, r] = np.einsum('ijk,j,k->i', dT_reshaped, V[:, r], W[:, r])
                gV[:, r] = np.einsum('ijk,i,k->j', dT_reshaped, U[:, r], W[:, r])
                gW[:, r] = np.einsum('ijk,i,j->k', dT_reshaped, U[:, r], V[:, r])

            # Momentum update
            vU = momentum * vU - lr * gU
            vV = momentum * vV - lr * gV
            vW = momentum * vW - lr * gW
            U += vU
            V += vV
            W += vW

        if best_err < 1e-6:
            break

    # Check if we found a good solution
    if best_factors is not None and best_err < 1e-6:
        U, V, W = best_factors
        # Round to half-integers
        U_r = round_to_half_int(U)
        V_r = round_to_half_int(V)
        W_r = round_to_half_int(W)

        # Verify
        if verify_decomposition(U_r, V_r, W_r, T, tol=1e-9):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    return None

# ---------------------------------------------------------------------------
# Main construct function
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    deadline = 120.0 - 5.0  # Leave 5s safety margin

    # Try to find a rank-13 decomposition
    T = build_target_tensor()

    # Only attempt search if we have enough time
    if time.time() - start_time < deadline - 30.0:
        result13 = search_rank13(T, time_budget=deadline - time.time() - 5.0)
        if result13 is not None:
            return result13

    # Fall back to rank-14
    return construct_rank14()
