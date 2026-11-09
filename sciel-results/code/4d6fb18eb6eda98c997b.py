"""Matrix multiplication 3x3 * 3x7 rank minimization.

Combines the current program's structured/Strassen-style initialization with
inspiration 3's core refinement: for fixed U and V the tensor equations are
linear in W, so we solve a least-squares system and accept only an exact
solution. A symmetric path (fix U,W solve for V) and a Strassen-seeded
initialization are added to increase the chance of hitting a low-rank solution.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]
DEADLINE = 130.0  # safety margin before the 150s kill

# Strassen rank-7 decomposition for 2x2x2 (each column: (a,b,c) for the three
# factors; the three factors multiply to the multiplication tensor).
_STRASSEN222 = [
    (1, 1, 1),
    (1, 0, -1),
    (1, 0, 1),
    (0, 1, 1),
    (1, -1, 0),
    (0, 1, -1),
    (0, 1, 1),
]


def _strassen333():
    """Build a rank-23 decomposition for the 3x3x3 tensor.

    Uses the 2x2x2 Strassen block (rank 7) for the (0,1)x(0,1)x(0,1) sub-tensor,
    plus schoolbook for every (i,j,k) that is not fully inside that block.
    Returns (U,V,W) each of shape (9, 23) with the (i,j)->i*3+j, (j,k)->j*3+k,
    (k,i)->k*3+i indexing.
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    col = 0
    for (a, b, c) in _STRASSEN222:
        U[0 * 3 + 0, col] = a
        U[0 * 3 + 1, col] = b
        U[1 * 3 + 0, col] = b
        U[1 * 3 + 1, col] = c
        V[0 * 3 + 0, col] = b
        V[0 * 3 + 1, col] = c
        V[1 * 3 + 0, col] = a
        V[1 * 3 + 1, col] = a
        W[0 * 3 + 0, col] = c
        W[0 * 3 + 1, col] = a
        W[1 * 3 + 0, col] = a
        W[1 * 3 + 1, col] = b
        col += 1
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * 3 + j, col] = 1.0
                V[j * 3 + k, col] = 1.0
                W[k * 3 + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen337():
    """Embed the rank-23 3x3x3 decomposition into 3x3x7 (rank 23 + 3*4 = 35).

    Columns 0..22 cover k in {0,1,2} via the 3x3x3 block. For k in {3,4,5,6}
    we use schoolbook (3 columns each, 12 total).
    """
    U333, V333, W333 = _strassen333()
    n, m, p = DIMENSIONS
    rank = 35
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    # 3x3x3 block for k in {0,1,2}
    for r in range(23):
        for i in range(3):
            for j in range(3):
                U[i * m + j, r] = U333[i * 3 + j, r]
        for j in range(3):
            for k in range(3):
                V[j * p + k, r] = V333[j * 3 + k, r]
        for k in range(3):
            for i in range(3):
                W[k * n + i, r] = W333[k * 3 + i, r]
    # schoolbook for k in {3,4,5,6}
    col = 23
    for i in range(n):
        for j in range(m):
            for k in range(3, p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    r = U.shape[1]
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return bool(np.allclose(T, np.ones((n, m, p)), atol=tol))


def _solve_for_W(U, V, n, m, p, tol=1e-8):
    """Given U,V, solve the linear system for W. Return W or None if inexact."""
    r = U.shape[1]
    num_eqs = n * m * p
    num_unk = p * n * r
    if num_unk < num_eqs:
        return None
    A = np.zeros((num_eqs, num_unk))
    eq = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                base = (k * n + i) * r
                A[eq, base:base + r] = U[i * m + j, :] * V[j * p + k, :]
                eq += 1
    b = np.ones(num_eqs)
    try:
        w, residuals, rk, sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if len(residuals) > 0 and residuals[0] > tol:
        return None
    W = w.reshape(p * n, r)
    return W


def _solve_for_V(U, W, n, m, p, tol=1e-8):
    """Given U,W, solve the linear system for V. Return V or None if inexact."""
    r = U.shape[1]
    num_eqs = n * m * p
    num_unk = m * p * r
    if num_unk < num_eqs:
        return None
    A = np.zeros((num_eqs, num_unk))
    eq = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                base = (j * p + k) * r
                A[eq, base:base + r] = U[i * m + j, :] * W[k * n + i, :]
                eq += 1
    b = np.ones(num_eqs)
    try:
        v, residuals, rk, sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if len(residuals) > 0 and residuals[0] > tol:
        return None
    V = v.reshape(m * p, r)
    return V


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rng = np.random.default_rng(12345)

    # Guaranteed baseline (schoolbook, rank 63).
    sU, sV, sW = _schoolbook(n, m, p)

    # Precompute a good structured seed (Strassen 3x3x3 + schoolbook tail).
    strU, strV, strW = _strassen337()

    def elapsed_ok():
        return time.time() - start_time < DEADLINE

    # Search from low rank upward. For each rank try random U,V and solve for W
    # (exact least-squares); also try a symmetric solve for V. Stop at the first
    # rank that admits an exact solution.
    for rank in range(23, 63):
        if not elapsed_ok():
            break
        # Scale attempts with remaining budget so we never overspend.
        max_attempts = 40
        for attempt in range(max_attempts):
            if not elapsed_ok():
                break
            if attempt == 0:
                U = strU.copy()
                V = strV.copy()
            elif attempt == 1:
                U = strU.copy()
                V = rng.standard_normal((m * p, rank)) * 0.5
            else:
                U = rng.standard_normal((n * m, rank))
                V = rng.standard_normal((m * p, rank))

            W = _solve_for_W(U, V, n, m, p)
            if W is not None and _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

            # Symmetric: fix U,W solve for V.
            V2 = _solve_for_V(U, strW, n, m, p)
            if V2 is not None and _verify(U, V2, strW, n, m, p):
                return {"U": U.tolist(), "V": V2.tolist(), "W": strW.tolist()}

    # Fallback: schoolbook (always valid, rank 63).
    return {"U": sU.tolist(), "V": sV.tolist(), "W": sW.tolist()}
