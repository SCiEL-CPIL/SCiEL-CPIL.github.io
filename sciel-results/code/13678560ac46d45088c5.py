"""Tensor rank decomposition for 2x4 x 4x6 matrix multiplication.

Strategy: model the M_{2,4,6} multiplication tensor as a Kronecker product of
a 2x2x2 (Strassen, rank 7) sub-tensor and a 1x2x3 sub-tensor. We search for a
rank-minimal exact decomposition of the 1x2x3 sub-problem (schoolbook rank 6),
then combine it with the Strassen 2x2x2 decomposition. This decouples the row
dimension and, if the sub-problem admits a rank-5 decomposition, yields a total
rank of 35 (vs the blockwise Strassen 42).
"""
import time
import sys
import random
from fractions import Fraction

import numpy as np

N, M, P = 2, 4, 6

DEADLINE = time.time() + 143  # safety margin under the 150s kill


# ---------------------------------------------------------------------------
# Exact verification of the full M_{2,4,6} tensor.
# ---------------------------------------------------------------------------
def _verify(U, V, W):
    r = U.shape[1]
    UF = [[Fraction(U[i, t]) for t in range(r)] for i in range(N * M)]
    VF = [[Fraction(V[i, t]) for t in range(r)] for i in range(M * P)]
    WF = [[Fraction(W[i, t]) for t in range(r)] for i in range(P * N)]
    for i in range(N):
        for k in range(P):
            for j1 in range(M):
                for j2 in range(M):
                    target = 1 if j1 == j2 else 0
                    s = Fraction(0)
                    for t in range(r):
                        s += UF[i * M + j1][t] * VF[j2 * P + k][t] * WF[k * N + i][t]
                    if s != target:
                        return False
    return True


# ---------------------------------------------------------------------------
# Kronecker-product factorisation.
# ---------------------------------------------------------------------------
def _kron_factor(U1, V1, W1, U2, V2, W2):
    """Kronecker product of two tensor decompositions."""
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    U = np.zeros((U1.shape[0] * U2.shape[0], r1 * r2))
    V = np.zeros((V1.shape[0] * V2.shape[0], r1 * r2))
    W = np.zeros((W1.shape[0] * W2.shape[0], r1 * r2))
    for t1 in range(r1):
        for t2 in range(r2):
            c = t1 * r2 + t2
            U[:, c] = np.kron(U1[:, t1], U2[:, t2])
            V[:, c] = np.kron(V1[:, t1], V2[:, t2])
            W[:, c] = np.kron(W1[:, t1], W2[:, t2])
    return U, V, W


# ---------------------------------------------------------------------------
# Strassen 2x2x2 decomposition (rank 7).
# ---------------------------------------------------------------------------
def _strassen_2x2x2():
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 0, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 1]),
    ]
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    for t in range(7):
        U[:, t] = terms[t][0]
        V[:, t] = terms[t][1]
        W[:, t] = terms[t][2]
    return U, V, W


# ---------------------------------------------------------------------------
# 1x2x3 sub-problem: find a rank-minimal exact decomposition.
#   U: (2, r), V: (3, r), W: (1, r)
#   condition: sum_t U[j,t]*V[k,t]*W[0,t] == delta_{jk}
# ---------------------------------------------------------------------------
def _sub_als(U, V, W):
    r = U.shape[1]
    # Solve U: for each j, sum_t U[j,t]*(V[k,t]*W[0,t]) = delta_{jk}
    U_new = U.copy()
    for j in range(2):
        A_mat = np.zeros((3, r))
        b = np.zeros(3)
        for k in range(3):
            A_mat[k, :] = V[k, :] * W[0, :]
            b[k] = 1.0 if k == j else 0.0
        sol, *_ = np.linalg.lstsq(A_mat, b, rcond=None)
        U_new[j, :] = sol
    # Solve V: for each k, sum_t V[k,t]*(U[j,t]*W[0,t]) = delta_{jk}
    V_new = V.copy()
    for k in range(3):
        A_mat = np.zeros((2, r))
        b = np.zeros(2)
        for j in range(2):
            A_mat[j, :] = U_new[j, :] * W[0, :]
            b[j] = 1.0 if j == k else 0.0
        sol, *_ = np.linalg.lstsq(A_mat, b, rcond=None)
        V_new[k, :] = sol
    # Solve W: for each t, W[0,t] = sum_jk U[j,t]*V[k,t]*delta_{jk} / (U[j,t]*V[k,t])
    W_new = W.copy()
    for t in range(r):
        num = 0.0
        den = 0.0
        for j in range(2):
            for k in range(3):
                if j == k:
                    num += U_new[j, t] * V_new[k, t]
                    den += U_new[j, t] * V_new[k, t]
        if abs(den) > 1e-12:
            W_new[0, t] = num / den
    return U_new, V_new, W_new


def _sub_residual(U, V, W):
    r = U.shape[1]
    err = 0.0
    for j in range(2):
        for k in range(3):
            target = 1.0 if j == k else 0.0
            s = 0.0
            for t in range(r):
                s += U[j, t] * V[k, t] * W[0, t]
            err += (s - target) ** 2
    return err


def _round_halfint(x):
    return np.round(x * 2) / 2


def _sub_verify(U, V, W):
    r = U.shape[1]
    UF = [[Fraction(U[i, t]) for t in range(r)] for i in range(2)]
    VF = [[Fraction(V[i, t]) for t in range(r)] for i in range(3)]
    WF = [[Fraction(W[i, t]) for t in range(r)] for i in range(1)]
    for j in range(2):
        for k in range(3):
            target = 1 if j == k else 0
            s = Fraction(0)
            for t in range(r):
                s += UF[j][t] * VF[k][t] * WF[0][t]
            if s != target:
                return False
    return True


def _sub_schoolbook(r):
    """Schoolbook decomposition of the 1x2x3 sub-problem, rank r >= 2."""
    U = np.zeros((2, r))
    V = np.zeros((3, r))
    W = np.zeros((1, r))
    for j in range(2):
        U[j, j] = 1.0
        V[j, j] = 1.0
        W[0, j] = 1.0
    # For j=2 (no corresponding row), leave columns 2..r-1 zero (they contribute nothing)
    return U, V, W


def _try_sub_rank(r, max_iters=200, n_restarts=5, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    for restart in range(n_restarts):
        if time.time() > DEADLINE - 5:
            break
        U = np.random.randn(2, r)
        V = np.random.randn(3, r)
        W = np.random.randn(1, r)
        for it in range(max_iters):
            if time.time() > DEADLINE - 5:
                break
            U, V, W = _sub_als(U, V, W)
            Ur, Vr, Wr = _round_halfint(U), _round_halfint(V), _round_halfint(W)
            if _sub_verify(Ur, Vr, Wr):
                return Ur, Vr, Wr
        Ur, Vr, Wr = _round_halfint(U), _round_halfint(V), _round_halfint(W)
        if _sub_verify(Ur, Vr, Wr):
            return Ur, Vr, Wr
    return None


# ---------------------------------------------------------------------------
# Blockwise Strassen baseline (rank 42).
# ---------------------------------------------------------------------------
def _strassen_baseline():
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 0, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 1]),
    ]
    rank = 42
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    for a_block in range(2):
        for b_block in range(3):
            for t in range(7):
                col = a_block * 21 + b_block * 7 + t
                U_t, V_t, W_t = terms[t]
                a_indices = [0, 1, 4, 5] if a_block == 0 else [2, 3, 6, 7]
                for i in range(4):
                    U[a_indices[i], col] = U_t[i]
                b_row_start = 0 if a_block == 0 else 2
                b_col_start = b_block * 2
                b_indices = [
                    b_row_start * 6 + b_col_start,
                    b_row_start * 6 + b_col_start + 1,
                    (b_row_start + 1) * 6 + b_col_start,
                    (b_row_start + 1) * 6 + b_col_start + 1,
                ]
                for i in range(4):
                    V[b_indices[i], col] = V_t[i]
                w_indices = [
                    2 * b_block,
                    2 * b_block + 1,
                    6 + 2 * b_block,
                    6 + 2 * b_block + 1,
                ]
                for i in range(4):
                    W[w_indices[i], col] = W_t[i]
    return U, V, W


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------
def construct():
    start = time.time()

    # Baseline: blockwise Strassen (rank 42).
    U, V, W = _strassen_baseline()
    if not _verify(U, V, W):
        # Schoolbook fallback (rank 48).
        rank = N * M * P
        U = np.zeros((N * M, rank))
        V = np.zeros((M * P, rank))
        W = np.zeros((P * N, rank))
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j, col] = 1
                    V[j * P + k, col] = 1
                    W[k * N + i, col] = 1
                    col += 1
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    best_rank = U.shape[1]
    best = (U, V, W)

    # Kronecker-product approach: Strassen 2x2x2 (rank 7) x 1x2x3 sub-problem.
    U1, V1, W1 = _strassen_2x2x2()

    # Try to find a rank-5 decomposition of the 1x2x3 sub-problem.
    sub5 = _try_sub_rank(5, max_iters=300, n_restarts=8, seed=42)
    if sub5 is not None and time.time() < DEADLINE - 10:
        U2, V2, W2 = sub5
        Uc, Vc, Wc = _kron_factor(U1, V1, W1, U2, V2, W2)
        if _verify(Uc, Vc, Wc):
            best_rank = Uc.shape[1]
            best = (Uc, Vc, Wc)
            return {'U': Uc.tolist(), 'V': Vc.tolist(), 'W': Wc.tolist()}

    # Fallback: use the schoolbook rank-6 sub-problem (gives total rank 42, same as baseline).
    U2, V2, W2 = _sub_schoolbook(6)
    Uc, Vc, Wc = _kron_factor(U1, V1, W1, U2, V2, W2)
    if _verify(Uc, Vc, Wc):
        if Uc.shape[1] < best_rank:
            best_rank = Uc.shape[1]
            best = (Uc, Vc, Wc)

    U, V, W = best
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
