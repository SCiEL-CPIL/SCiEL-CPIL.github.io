"""2x5x6 matrix multiplication tensor decomposition.

Strategy:
  1. Build a verified block-Strassen rank-54 decomposition (j split 2+2+1, k split 2+2+2).
  2. Verify it exhaustively (valid triples must be 1, invalid triples must be 0).
  3. Systematically attempt term elimination to reduce rank below 54.
  4. Fall back to schoolbook rank-60 if everything fails.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill


def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Exhaustive verification: valid triples -> 1, invalid triples -> 0."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False

    # Valid triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(s - 1.0) > tol:
                    return False

    # Invalid triples: sum_r U[y,r]*V[z,r]*W[x,r] must be 0
    # y indexes (i2,j), z indexes (j2,k2), x indexes (k,i)
    # They are consistent iff i==i2, j==j2, k==k2
    for x in range(p * n):
        i = x % n
        k = x // n
        for y in range(n * m):
            j = y % m
            i2 = y // m
            for z in range(m * p):
                j2 = z // p
                k2 = z % p
                if i == i2 and j == j2 and k == k2:
                    continue
                s = float(np.dot(U[y], V[z] * W[x]))
                if abs(s) > tol:
                    return False
    return True


def _check_half_integer(U, V, W):
    """Check all entries are integer or half-integer."""
    for mat in [U, V, W]:
        if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-12):
            return False
    return True


def _build_54_rank(n, m, p):
    """Block-Strassen rank-54 decomposition.

    j split: (0,1), (2,3), (4).  k split: (0,1), (2,3), (4,5).
    Each 2x2x2 block uses the 7-term Strassen scheme.  j=4 uses schoolbook (rank 12).
    Total: 2 * 3 * 7 + 12 = 54.
    """
    rank = 54
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]

    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # Standard Strassen 2x2x2, local indices 0,1 -> global j0,j1 / k0,k1
            # P1 = (A00+A11)(B00+B11)
            U[0 * m + j0, col] += 1
            U[1 * m + j1, col] += 1
            V[j0 * p + k0, col] += 1
            V[j1 * p + k1, col] += 1
            W[k0 * n + 0, col] += 1
            W[k1 * n + 1, col] += 1
            col += 1

            # P2 = (A10+A11)B00
            U[1 * m + j0, col] += 1
            U[1 * m + j1, col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + 1, col] += 1
            W[k1 * n + 1, col] -= 1
            col += 1

            # P3 = A00(B01-B11)
            U[0 * m + j0, col] += 1
            V[j0 * p + k1, col] += 1
            V[j1 * p + k1, col] -= 1
            W[k1 * n + 0, col] += 1
            W[k1 * n + 1, col] += 1
            col += 1

            # P4 = A11(B10-B00)
            U[1 * m + j1, col] += 1
            V[j1 * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + 0, col] += 1
            W[k0 * n + 1, col] += 1
            col += 1

            # P5 = (A00+A01)B11
            U[0 * m + j0, col] += 1
            U[0 * m + j1, col] += 1
            V[j1 * p + k1, col] += 1
            W[k0 * n + 0, col] -= 1
            W[k1 * n + 0, col] += 1
            col += 1

            # P6 = (A10-A00)(B00+B10)
            U[1 * m + j0, col] += 1
            U[0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + k1, col] += 1
            W[k1 * n + 1, col] += 1
            col += 1

            # P7 = (A01-A11)(B01+B11)
            U[0 * m + j1, col] += 1
            U[1 * m + j1, col] -= 1
            V[j1 * p + k0, col] += 1
            V[j1 * p + k1, col] += 1
            W[k0 * n + 0, col] += 1
            col += 1

    # Remaining j=4: schoolbook rank 12
    for k in range(p):
        U[0 * m + 4, col] += 1
        V[4 * p + k, col] += 1
        W[k * n + 0, col] += 1
        col += 1
        U[1 * m + 4, col] += 1
        V[4 * p + k, col] += 1
        W[k * n + 1, col] += 1
        col += 1

    assert col == rank, f"Expected col={rank}, got {col}"
    return U, V, W


def _try_eliminate_terms(U, V, W, n, m, p, time_budget=10.0):
    """Attempt to reduce rank by removing redundant terms.

    A term r is redundant if its tensor contribution (the rank-1 tensor
    U[:,r] o V[:,r] o W[:,r]) lies in the span of the other terms' contributions.
    We test this by checking whether removing term r still leaves a valid
    decomposition (i.e., the residual can be absorbed). Concretely, we drop term r
    and check if the remaining decomposition still satisfies the identity.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    start = time.time()

    # Greedy elimination: repeatedly try to drop a term and re-verify.
    # To keep it tractable, we test each term once per pass.
    while _time_left() > 15.0 and (time.time() - start) < time_budget:
        if rank <= 1:
            break
        removed = False
        for r in range(rank):
            if _time_left() < 15.0:
                break
            # Try dropping term r
            U_t = np.delete(U, r, axis=1)
            V_t = np.delete(V, r, axis=1)
            W_t = np.delete(W, r, axis=1)
            if _verify_decomposition(U_t, V_t, W_t, n, m, p, tol=1e-9):
                U, V, W = U_t, V_t, W_t
                rank -= 1
                removed = True
                break
        if not removed:
            break
    return U, V, W


def _schoolbook():
    """Schoolbook decomposition, rank = n*m*p = 60."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}


def _to_lists(U, V, W):
    return {
        "U": np.asarray(U, dtype=float).tolist(),
        "V": np.asarray(V, dtype=float).tolist(),
        "W": np.asarray(W, dtype=float).tolist(),
    }


def construct():
    n, m, p = DIMENSIONS

    # Primary: block-Strassen rank-54
    try:
        U, V, W = _build_54_rank(n, m, p)
        if _verify_decomposition(U, V, W, n, m, p) and _check_half_integer(U, V, W):
            best = _to_lists(U, V, W)
            best_rank = U.shape[1]

            # Attempt rank reduction via term elimination
            if _time_left() > 20.0:
                U2, V2, W2 = _try_eliminate_terms(U, V, W, n, m, p, time_budget=10.0)
                if _verify_decomposition(U2, V2, W2, n, m, p) and _check_half_integer(U2, V2, W2):
                    if U2.shape[1] < best_rank:
                        best = _to_lists(U2, V2, W2)
                        best_rank = U2.shape[1]
            return best
    except Exception:
        pass

    # Fallback: schoolbook rank-60
    sb = _schoolbook()
    return _to_lists(sb['U'], sb['V'], sb['W'])

# EVOLVE-BLOCK-END
