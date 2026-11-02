import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 2, 9]
_START_TIME = time.time()
_TIMEOUT = 135  # safety margin before 150s kill


def _build_rank_32(n, m, p):
    """
    Build a rank-32 decomposition using Strassen for 4 blocks of 2x2x2
    and schoolbook for 1 block of 2x2x1.
    """
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

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

    def add_standard(k):
        nonlocal col
        # Standard 2x2x1 multiplication
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

    # 4 blocks of 2 columns using Strassen (rank 7 each)
    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)

    # 1 block of 1 column using standard (rank 4)
    add_standard(8)

    return U, V, W


def _validate(U, V, W, n, m, p):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_recon = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    return np.allclose(T, T_recon, atol=1e-10)


def _try_remove_terms(U, V, W, n, m, p, max_pairs=5000, max_triples=200):
    """Try to remove terms from the decomposition via single, pair, and triple removal."""
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 15:
        return None

    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def check_removal(indices_to_remove):
        mask = [i for i in range(rank) if i not in indices_to_remove]
        U_sub = Ua[:, mask]
        V_sub = Va[:, mask]
        W_sub = Wa[:, mask]
        T_recon = np.einsum('ir,jr,kr->ijk', U_sub, V_sub, W_sub)
        if np.allclose(T, T_recon, atol=1e-10):
            new_rank = rank - len(indices_to_remove)
            return (
                [[U_sub[i][j] for j in range(new_rank)] for i in range(n * m)],
                [[V_sub[i][j] for j in range(new_rank)] for i in range(m * p)],
                [[W_sub[i][j] for j in range(new_rank)] for i in range(p * n)],
            )
        return None

    # Try removing each single term
    for r in range(rank):
        if time.time() - _START_TIME > _TIMEOUT - 10:
            return None
        result = check_removal([r])
        if result is not None:
            return result

    # Try removing pairs of terms
    pairs_checked = 0
    for r1 in range(rank):
        for r2 in range(r1 + 1, rank):
            if time.time() - _START_TIME > _TIMEOUT - 10:
                return None
            if pairs_checked >= max_pairs:
                break
            pairs_checked += 1
            result = check_removal([r1, r2])
            if result is not None:
                return result
        if pairs_checked >= max_pairs:
            break

    # Try removing triples of terms (limited search)
    triples_checked = 0
    for r1 in range(rank):
        for r2 in range(r1 + 1, rank):
            for r3 in range(r2 + 1, rank):
                if time.time() - _START_TIME > _TIMEOUT - 10:
                    return None
                if triples_checked >= max_triples:
                    break
                triples_checked += 1
                result = check_removal([r1, r2, r3])
                if result is not None:
                    return result
            if triples_checked >= max_triples:
                break
        if triples_checked >= max_triples:
            break

    return None


def _polish_decomposition(U, V, W, n, m, p):
    """
    Attempt to reduce rank by finding linear dependencies in the U matrix.
    Since U has only 4 rows, there are structural dependencies.
    We try to find a lower-rank representation by combining terms.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 20:
        return None

    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # The U matrix is 4 x rank. If rank > 4, columns of U are linearly dependent.
    # We can try to find a basis for the column space of U and express all columns
    # in terms of this basis, potentially reducing the effective rank.
    
    # Compute rank of U
    U_rank = np.linalg.matrix_rank(Ua, tol=1e-10)
    
    if U_rank >= rank:
        return None  # No reduction possible via U dependencies

    # Find a set of linearly independent columns of U
    pivot_cols = []
    for c in range(rank):
        col = Ua[:, c]
        if np.linalg.norm(col) < 1e-10:
            continue
        # Check if this column is in the span of previous pivot columns
        if not pivot_cols:
            pivot_cols.append(c)
        else:
            P = Ua[:, pivot_cols]
            # Solve P @ x = col
            try:
                x, residuals, _, _ = np.linalg.lstsq(P, col, rcond=None)
                if np.linalg.norm(P @ x - col) < 1e-10:
                    continue  # Dependent, skip
            except:
                pass
            pivot_cols.append(c)

    if len(pivot_cols) >= rank:
        return None

    # We have len(pivot_cols) < rank independent columns
    # The non-pivot columns can be expressed as linear combinations
    # This doesn't directly reduce rank, but tells us about structure
    
    # Alternative approach: try to find if any subset of terms can be
    # replaced by a single term with adjusted coefficients
    # This is complex, so we fall back to the removal search
    
    return None


def construct():
    n, m, p = DIMENSIONS

    # Build rank-32 decomposition
    U, V, W = _build_rank_32(n, m, p)

    # Validate
    if not _validate(U, V, W, n, m, p):
        # Fallback: schoolbook rank 36
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

    # Try to reduce rank via term removal
    reduced = _try_remove_terms(U, V, W, n, m, p)
    if reduced is not None:
        U2, V2, W2 = reduced
        if _validate(U2, V2, W2, n, m, p):
            return {"U": U2, "V": V2, "W": W2}

    # Try polish (structural reduction)
    polished = _polish_decomposition(U, V, W, n, m, p)
    if polished is not None:
        U3, V3, W3 = polished
        if _validate(U3, V3, W3, n, m, p):
            return {"U": U3, "V": V3, "W": W3}

    return {"U": U, "V": V, "W": W}
