"""Rewritten search for a rank-minimal exact decomposition of 3x4 * 4x4 multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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


def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    Uf = [[Fraction(U[i][c]) for c in range(rank)] for i in range(len(U))]
    Vf = [[Fraction(V[i][c]) for c in range(rank)] for i in range(len(V))]
    Wf = [[Fraction(W[i][c]) for c in range(rank)] for i in range(len(W))]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Uf[i * m + j][c] * Vf[j * p + k][c] * Wf[k * n + i][c]
                if s != 1:
                    return False
    return True


def _snap_half(x, tol=1e-10):
    v = round(x * 2) / 2
    return v if abs(x - v) < tol else x


def _snap_matrix(M, tol=1e-10):
    return [[_snap_half(x, tol) for x in row] for row in M]


def _try_random_uv(n, m, p, rank, max_time=5.0, seed=None):
    """
    Randomly draw integer U,V with entries in {-2,-1,0,1,2}, solve for W
    via per-(i,k) least-squares, snap to half-integers, and verify exactly.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    t0 = time.time()
    deadline = t0 + max_time
    nU, nV, nW = n * m, m * p, p * n
    while time.time() < deadline - 1.0:
        U_np = np.random.randint(-2, 3, (nU, rank))
        V_np = np.random.randint(-2, 3, (nV, rank))
        W_np = np.zeros((nW, rank))
        ok = True
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        A[j, c] = U_np[i * m + j, c] * V_np[j * p + k, c]
                try:
                    W_ik, res, rA, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    ok = False
                    break
                if np.linalg.norm(A @ W_ik - np.ones(m)) > 1e-9:
                    ok = False
                    break
                W_np[k * n + i, :] = W_ik
            if not ok:
                break
        if not ok:
            continue
        Us = _snap_matrix(U_np.tolist())
        Vs = _snap_matrix(V_np.tolist())
        Ws = _snap_matrix(W_np.tolist())
        if _verify(Us, Vs, Ws, n, m, p):
            return Us, Vs, Ws, rank
    return None


def _try_structured(n, m, p, rank, max_time=5.0):
    """
    Try structured forms: U and V have a block-structured pattern with a few
    random sign flips, then solve for W.
    """
    t0 = time.time()
    deadline = t0 + max_time
    nU, nV, nW = n * m, m * p, p * n
    attempts = 0
    while time.time() < deadline - 1.0:
        attempts += 1
        # Random signs for U and V entries, magnitude 1
        U_np = np.zeros((nU, rank))
        V_np = np.zeros((nV, rank))
        for c in range(rank):
            # Each column of U is a random ±1 vector
            U_np[:, c] = np.random.choice([-1, 1], nU)
            V_np[:, c] = np.random.choice([-1, 1], nV)
        W_np = np.zeros((nW, rank))
        ok = True
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        A[j, c] = U_np[i * m + j, c] * V_np[j * p + k, c]
                try:
                    W_ik, res, rA, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    ok = False
                    break
                if np.linalg.norm(A @ W_ik - np.ones(m)) > 1e-9:
                    ok = False
                    break
                W_np[k * n + i, :] = W_ik
            if not ok:
                break
        if not ok:
            continue
        Us = _snap_matrix(U_np.tolist())
        Vs = _snap_matrix(V_np.tolist())
        Ws = _snap_matrix(W_np.tolist())
        if _verify(Us, Vs, Ws, n, m, p):
            return Us, Vs, Ws, rank
    return None


def _try_perturb_schoolbook(n, m, p, target_rank, max_time=5.0):
    """
    Start from schoolbook (rank 48) and try to zero out one column at a time
    by solving a constrained system. This is a heuristic.
    """
    t0 = time.time()
    deadline = t0 + max_time
    nU, nV, nW = n * m, m * p, p * n
    # Schoolbook
    U, V, W = _schoolbook(n, m, p)
    current_rank = n * m * p
    # Try to remove columns one by one
    for col_to_remove in range(target_rank, current_rank):
        if time.time() > deadline - 2.0:
            break
        # Remove column col_to_remove
        U_new = [row[:col_to_remove] + row[col_to_remove + 1:] for row in U]
        V_new = [row[:col_to_remove] + row[col_to_remove + 1:] for row in V]
        W_new = [row[:col_to_remove] + row[col_to_remove + 1:] for row in W]
        new_rank = col_to_remove
        if _verify(U_new, V_new, W_new, n, m, p):
            return U_new, V_new, W_new, new_rank
        # If not valid, try a small perturbation: add a random column to compensate
        # This is very heuristic; skip for now
    return None


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin
    n, m, p = DIMENSIONS

    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U, V, W, n * m * p

    # Try rank 47
    if time.time() < deadline - 30.0:
        res = _try_random_uv(n, m, p, best_rank - 1, max_time=15.0, seed=42)
        if res:
            best_U, best_V, best_W, best_rank = res

    # Try rank 46
    if best_rank > 46 and time.time() < deadline - 20.0:
        res = _try_random_uv(n, m, p, 46, max_time=10.0, seed=123)
        if res:
            best_U, best_V, best_W, best_rank = res

    # Try structured search for rank 47
    if best_rank > 47 and time.time() < deadline - 15.0:
        res = _try_structured(n, m, p, 47, max_time=10.0)
        if res:
            best_U, best_V, best_W, best_rank = res

    # Try structured search for rank 46
    if best_rank > 46 and time.time() < deadline - 10.0:
        res = _try_structured(n, m, p, 46, max_time=8.0)
        if res:
            best_U, best_V, best_W, best_rank = res

    # Try perturbation of schoolbook
    if best_rank > 47 and time.time() < deadline - 5.0:
        res = _try_perturb_schoolbook(n, m, p, 47, max_time=5.0)
        if res:
            best_U, best_V, best_W, best_rank = res

    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = U, V, W
        best_rank = n * m * p

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
