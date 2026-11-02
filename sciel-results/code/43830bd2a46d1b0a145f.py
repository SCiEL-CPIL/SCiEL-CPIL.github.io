"""Tensor decomposition for 2x2 x 2x9 matrix multiplication."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIMEOUT = 130  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _build_tensor(n, m, p):
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _validate(U, V, W, n, m, p, atol=1e-9):
    """Check if U,V,W exactly decompose the multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = _build_tensor(n, m, p)
    T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T, T_recon, atol=atol)


def _build_rank_32(n, m, p):
    """
    Build a rank-32 decomposition using Strassen for 4 blocks of 2x2x2
    and schoolbook for 1 block of 2x2x1.

    For n=2, m=2, p=9:
    - U shape: (n*m, rank) = (4, rank)
    - V shape: (m*p, rank) = (18, rank)
    - W shape: (p*n, rank) = (18, rank)

    U index: i*m + j for i in {0,1}, j in {0,1}
      U[0] = A00, U[1] = A01, U[2] = A10, U[3] = A11

    V index: j*p + k for j in {0,1}, k in 0..8
      V[j*9 + k] = B[j, k]

    W index: k*n + i for k in 0..8, i in {0,1}
      W[k*2 + i] = C[i, k]  (row i, column k of output)
    """
    rank = 32
    U = [[0] * rank for _ in range(n * m)]      # 4 x 32
    V = [[0] * rank for _ in range(m * p)]      # 18 x 32
    W = [[0] * rank for _ in range(p * n)]      # 18 x 32

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        # M1: (A00+A11)*(B00+B11)
        U[0][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

        # M2: (A10+A11)*B00
        U[2][col] = 1
        U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        W[k1*n + 1][col] = -1
        col += 1

        # M3: A00*(B10-B00)
        U[0][col] = 1
        V[0*p + k1][col] = 1
        V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

        # M4: A11*(B01-B11)
        U[3][col] = 1
        V[1*p + k0][col] = 1
        V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1

        # M5: (A00+A01)*B11
        U[0][col] = 1
        U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1
        W[k1*n + 0][col] = 1
        col += 1

        # M6: (A10-A00)*(B00+B01)
        U[2][col] = 1
        U[0][col] = -1
        V[0*p + k0][col] = 1
        V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1

        # M7: (A01-A11)*(B10+B11)
        U[1][col] = 1
        U[3][col] = -1
        V[1*p + k0][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1

        U[1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1

        U[2][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

        U[3][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    # 4 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7)
    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    # Schoolbook for column 8
    add_trivial(8)

    return U, V, W


def _try_remove_terms(U, V, W, n, m, p, max_pair_removals=100):
    """
    Try to remove terms from the decomposition if they are linearly dependent.
    First tries single-term removal, then pair-term removal.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 10:
        return None

    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]

    T = _build_tensor(n, m, p)

    # Try removing each single term
    for r in range(rank):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        mask = [i for i in range(rank) if i != r]
        U_sub = Ua[:, mask]
        V_sub = Va[:, mask]
        W_sub = Wa[:, mask]
        T_recon = np.einsum('ir,jr,kr->ijk', U_sub, V_sub, W_sub)
        if np.allclose(T, T_recon, atol=1e-10):
            new_rank = rank - 1
            return (
                [[U_sub[i][j] for j in range(new_rank)] for i in range(n * m)],
                [[V_sub[i][j] for j in range(new_rank)] for i in range(m * p)],
                [[W_sub[i][j] for j in range(new_rank)] for i in range(p * n)],
            )

    # Try removing pairs of terms (limited to max_pair_removals checks for time)
    pairs_checked = 0
    for r1 in range(rank):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        for r2 in range(r1 + 1, rank):
            if pairs_checked >= max_pair_removals:
                return None
            pairs_checked += 1
            mask = [i for i in range(rank) if i != r1 and i != r2]
            U_sub = Ua[:, mask]
            V_sub = Va[:, mask]
            W_sub = Wa[:, mask]
            T_recon = np.einsum('ir,jr,kr->ijk', U_sub, V_sub, W_sub)
            if np.allclose(T, T_recon, atol=1e-10):
                new_rank = rank - 2
                return (
                    [[U_sub[i][j] for j in range(new_rank)] for i in range(n * m)],
                    [[V_sub[i][j] for j in range(new_rank)] for i in range(m * p)],
                    [[W_sub[i][j] for j in range(new_rank)] for i in range(p * n)],
                )

    return None


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS

    best = None
    best_rank = None

    # Build rank-32 decomposition using Strassen + schoolbook
    try:
        U, V, W = _build_rank_32(n, m, p)
        if _validate(U, V, W, n, m, p):
            rank = len(U[0])
            if best_rank is None or rank < best_rank:
                best = (U, V, W)
                best_rank = rank
    except Exception:
        pass

    # Try to reduce rank by removing linearly dependent terms
    if best is not None:
        U, V, W = best
        try:
            reduced = _try_remove_terms(U, V, W, n, m, p)
            if reduced is not None:
                U2, V2, W2 = reduced
                if _validate(U2, V2, W2, n, m, p):
                    rank2 = len(U2[0])
                    if best_rank is None or rank2 < best_rank:
                        best = (U2, V2, W2)
                        best_rank = rank2
        except Exception:
            pass

    if best is not None:
        U, V, W = best
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
