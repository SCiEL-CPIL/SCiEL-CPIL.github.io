import numpy as np
import time
import random

DIMENSIONS = [2, 2, 9]
_START_TIME = time.time()
_TIMEOUT = 140  # safety margin before 150s kill


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

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)

    return U, V, W


def _validate(U, V, W, n, m, p):
    """Check if the decomposition exactly decomposes the multiplication tensor."""
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


def _try_remove_terms(U, V, W, n, m, p):
    """Try to remove terms from the decomposition if they are linearly dependent."""
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 10:
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

    T_full = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    
    T_terms = np.zeros((rank, n * m, m * p, p * n), dtype=float)
    for r in range(rank):
        T_terms[r] = np.einsum('i,j,k->ijk', Ua[:, r], Va[:, r], Wa[:, r])

    if not np.allclose(T, T_full, atol=1e-10):
        return None

    def _check_removal(indices_to_remove):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        T_recon = T_full.copy()
        for idx in indices_to_remove:
            T_recon -= T_terms[idx]
        if np.allclose(T, T_recon, atol=1e-10):
            mask = [i for i in range(rank) if i not in indices_to_remove]
            new_rank = len(mask)
            return (
                [[Ua[i][j] for j in mask] for i in range(n * m)],
                [[Va[i][j] for j in mask] for i in range(m * p)],
                [[Wa[i][j] for j in mask] for i in range(p * n)],
            )
        return None

    for r in range(rank):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        result = _check_removal((r,))
        if result is not None:
            return result

    for r1 in range(rank):
        if time.time() - _START_TIME > _TIMEOUT - 5:
            return None
        for r2 in range(r1 + 1, rank):
            result = _check_removal((r1, r2))
            if result is not None:
                return result

    if time.time() - _START_TIME < _TIMEOUT - 15:
        for r1 in range(rank):
            if time.time() - _START_TIME > _TIMEOUT - 5:
                return None
            for r2 in range(r1 + 1, rank):
                if time.time() - _START_TIME > _TIMEOUT - 5:
                    return None
                for r3 in range(r2 + 1, rank):
                    if time.time() - _START_TIME > _TIMEOUT - 5:
                        return None
                    result = _check_removal((r1, r2, r3))
                    if result is not None:
                        return result

    return None


def _search_rank_lower(n, m, p, U0, V0, W0):
    """
    Search for a lower-rank decomposition using SVD null space analysis.
    If there's a linear dependency among the terms, we can reduce the rank.
    """
    elapsed = time.time() - _START_TIME
    if elapsed > _TIMEOUT - 20:
        return None

    Ua = np.array(U0, dtype=float)
    Va = np.array(V0, dtype=float)
    Wa = np.array(W0, dtype=float)
    rank = Ua.shape[1]

    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_full = np.einsum('ir,jr,kr->ijk', Ua, Va, Wa)
    if not np.allclose(T, T_full, atol=1e-10):
        return None

    # Flatten each term tensor to a vector
    T_flat = np.zeros((rank, n*m*m*p*p*n), dtype=float)
    for r in range(rank):
        T_flat[r] = np.einsum('i,j,k->ijk', Ua[:, r], Va[:, r], Wa[:, r]).flatten()
    
    # Find null space via SVD
    try:
        if time.time() - _START_TIME > _TIMEOUT - 10:
            return None
        U_svd, s, Vt = np.linalg.svd(T_flat, full_matrices=False)
        
        # Use multiple tolerances to find potential null space vectors
        for tol in [1e-10, 1e-8, 1e-6, 1e-4]:
            if time.time() - _START_TIME > _TIMEOUT - 5:
                return None
            num_zero = sum(1 for si in s if si < tol)
            if num_zero == 0:
                continue
            
            null_vecs = Vt[rank - num_zero:].T  # shape (rank, num_zero)
            
            for nv in null_vecs:
                if time.time() - _START_TIME > _TIMEOUT - 5:
                    return None
                
                # Strategy 1: eliminate terms where nv[r] is close to 0
                zero_terms = set(r for r in range(rank) if abs(nv[r]) < 1e-6)
                if len(zero_terms) > 0:
                    remaining = [r for r in range(rank) if r not in zero_terms]
                    U_new = Ua[:, remaining]
                    V_new = Va[:, remaining]
                    W_new = Wa[:, remaining]
                    T_check = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
                    if np.allclose(T, T_check, atol=1e-8):
                        return (U_new.tolist(), V_new.tolist(), W_new.tolist())
                
                # Strategy 2: eliminate the term with largest |nv[r]|
                max_idx = np.argmax(np.abs(nv))
                if abs(nv[max_idx]) > 1e-6:
                    new_mask = []
                    new_coeffs = []
                    for r in range(rank):
                        if r == max_idx:
                            continue
                        c = 1.0 - nv[r] / nv[max_idx]
                        if abs(c) > 1e-6:
                            new_mask.append(r)
                            new_coeffs.append(c)
                    
                    if len(new_mask) < rank:
                        U_new = Ua[:, new_mask] * np.array(new_coeffs)[np.newaxis, :]
                        V_new = Va[:, new_mask]
                        W_new = Wa[:, new_mask]
                        T_check = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
                        if np.allclose(T, T_check, atol=1e-8):
                            return (U_new.tolist(), V_new.tolist(), W_new.tolist())
    except Exception:
        pass

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

    # Try to find lower rank via linear dependency search
    if time.time() - _START_TIME < _TIMEOUT - 15:
        r_lower = _search_rank_lower(n, m, p, U, V, W)
        if r_lower is not None:
            U3, V3, W3 = r_lower
            if _validate(U3, V3, W3, n, m, p):
                return {"U": U3, "V": V3, "W": W3}

    return {"U": U, "V": V, "W": W}
