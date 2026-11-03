"""3x3x3 matrix multiplication tensor decomposition.

Strategy: start from the known rank-26 decomposition (Strassen 2x2 core
rank-7 + 19 pure border spikes), then attempt a structured algebraic
elimination to lower the rank toward 23 by absorbing border contributions
into combined core+border terms, gated by exact verification.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]
_start_time = time.time()
DEADLINE = 145.0


def _remaining():
    return DEADLINE - time.time()


def _reconstruct(U, V, W):
    """Reconstruct the matmul tensor T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    n, m, p = 3, 3, 3
    Ur = U.reshape(n, m, -1)
    Vr = V.reshape(m, p, -1)
    Wr = W.reshape(p, n, -1)
    return np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)


def _verify(U, V, W, tol=1e-9):
    T = _reconstruct(U, V, W)
    return np.allclose(T, np.ones((3, 3, 3)), atol=tol)


def _snap_half_int(arr):
    return np.round(np.asarray(arr, dtype=float) * 2) / 2.0


def _build_rank26():
    """Strassen 2x2 core (rank 7) + 19 border spikes = rank 26."""
    n, m, p = 3, 3, 3
    R = 26
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    def iu(i, j):
        return i * m + j

    def iv(j, k):
        return j * p + k

    def iw(k, i):
        return k * n + i

    # ---- Strassen 2x2 core (i,j,k in {0,1}) ----
    # m1 = (a00+a11)(b00+b11) -> c00 += m1, c11 += m1
    r = 0
    U[iu(0, 0), r] = 1; U[iu(1, 1), r] = 1
    V[iv(0, 0), r] = 1; V[iv(1, 1), r] = 1
    W[iw(0, 0), r] = 1; W[iw(1, 1), r] = 1
    # m2 = (a10+a11)b00 -> c10 += m2, c11 -= m2
    r = 1
    U[iu(1, 0), r] = 1; U[iu(1, 1), r] = 1
    V[iv(0, 0), r] = 1
    W[iw(0, 1), r] = 1; W[iw(1, 1), r] = -1
    # m3 = a00(b01-b11) -> c01 += m3, c11 += m3
    r = 2
    U[iu(0, 0), r] = 1
    V[iv(0, 1), r] = 1; V[iv(1, 1), r] = -1
    W[iw(1, 0), r] = 1; W[iw(1, 1), r] = 1
    # m4 = a11(b10-b00) -> c00 += m4, c10 += m4
    r = 3
    U[iu(1, 1), r] = 1
    V[iv(1, 0), r] = 1; V[iv(0, 0), r] = -1
    W[iw(0, 0), r] = 1; W[iw(0, 1), r] = 1
    # m5 = (a00+a01)b11 -> c00 -= m5, c01 += m5
    r = 4
    U[iu(0, 0), r] = 1; U[iu(0, 1), r] = 1
    V[iv(1, 1), r] = 1
    W[iw(0, 0), r] = -1; W[iw(1, 0), r] = 1
    # m6 = (a10-a00)(b00+b01) -> c11 += m6
    r = 5
    U[iu(1, 0), r] = 1; U[iu(0, 0), r] = -1
    V[iv(0, 0), r] = 1; V[iv(0, 1), r] = 1
    W[iw(1, 1), r] = 1
    # m7 = (a01-a11)(b10+b11) -> c00 += m7
    r = 6
    U[iu(0, 1), r] = 1; U[iu(1, 1), r] = -1
    V[iv(1, 0), r] = 1; V[iv(1, 1), r] = 1
    W[iw(0, 0), r] = 1

    # ---- Border spikes (19 terms: at least one index == 2) ----
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[iu(i, j), col] = 1
                V[iv(j, k), col] = 1
                W[iw(k, i), col] = 1
                col += 1
    assert col == R, (col, R)
    return U, V, W


def _build_rank27():
    """Schoolbook rank-27 fallback."""
    n, m, p = 3, 3, 3
    R = 27
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    def iu(i, j):
        return i * m + j

    def iv(j, k):
        return j * p + k

    def iw(k, i):
        return k * n + i

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[iu(i, j), col] = 1
                V[iv(j, k), col] = 1
                W[iw(k, i), col] = 1
                col += 1
    return U, V, W


def _try_eliminate(U26, V26, W26):
    """Attempt to reduce the rank-26 decomposition by absorbing border spikes
    into combined core+border terms.

    A border spike s covers exactly one tensor entry (i2,j2,k2) with index 2.
    If we add a border-contribution to an existing core term r (which already
    covers core entries), we can drop spike s, reducing the rank by 1. The
    requirement is that the core entries of r change in a way still representable
    by the 7-term Strassen subspace. We test candidate absorptions and verify.
    """
    n, m, p = 3, 3, 3

    def iu(i, j):
        return i * m + j

    def iv(j, k):
        return j * p + k

    def iw(k, i):
        return k * n + i

    R = 26
    # Identify which columns are core (0..6) and which are border spikes (7..25)
    # Map each border spike column to its (i,j,k)
    spike_info = {}
    for col in range(7, R):
        # find the unique (i,j,k) with a 1 in this column
        found = None
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if U26[iu(i, j), col] != 0 and V26[iv(j, k), col] != 0 and W26[iw(k, i), col] != 0:
                        found = (i, j, k)
        if found is not None:
            spike_info[col] = found

    # For each core term r and each border spike s, try: move spike s's border
    # contribution into core term r. This means:
    #   - in term r, set U[i2,j2]+=1, V[j2,k2]+=1, W[k2,i2]+=1 (add border part)
    #   - remove spike column s entirely
    # The core entries of term r are now perturbed. We check if the resulting
    # tensor still equals the target.
    #
    # We do this greedily: try each (r, s) pair, verify, keep if valid.

    # Work on copies
    U = U26.copy()
    V = V26.copy()
    W = W26.copy()
    active = list(range(R))  # active columns

    # Build target
    target = np.ones((3, 3, 3))

    improved = True
    while improved and _remaining() > 10:
        improved = False
        # current reconstruction
        # Try each active core term r (0..6) and each active spike s
        active_core = [c for c in active if c <= 6]
        active_spikes = [c for c in active if c in spike_info]
        for r in active_core:
            if _remaining() < 5:
                break
            for s in active_spikes:
                if _remaining() < 2:
                    break
                i2, j2, k2 = spike_info[s]
                # Try absorbing spike s into core term r
                U[iu(i2, j2), r] += 1
                V[iv(j2, k2), r] += 1
                W[iw(k2, i2), r] += 1
                # remove spike s
                U[:, s] = 0
                V[:, s] = 0
                W[:, s] = 0
                # verify
                if _verify(U, V, W):
                    active.remove(s)
                    improved = True
                    break
                else:
                    # revert
                    U[iu(i2, j2), r] -= 1
                    V[iv(j2, k2), r] -= 1
                    W[iw(k2, i2), r] -= 1
                    U[:, s] = U26[:, s]
                    V[:, s] = V26[:, s]
                    W[:, s] = W26[:, s]
            if improved:
                break
        # If no single spike absorbed, try creating a NEW combined term that
        # covers a border entry plus a core entry, replacing a spike + part of core.
        # This is harder; skip for now (the single-absorption loop is the main win).

    # Trim to active columns
    if active:
        U = U[:, active]
        V = V[:, active]
        W = W[:, active]
    if _verify(U, V, W):
        return U, V, W
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 matmul.

    Returns {'U','V','W'} with shapes (9, rank) each, integer/half-integer
    entries exactly decomposing the matmul tensor. Lower rank = better score.
    """
    U26, V26, W26 = _build_rank26()

    # Sanity: the rank-26 base must be valid
    if not _verify(U26, V26, W26):
        U27, V27, W27 = _build_rank27()
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}

    # Attempt structured elimination to lower the rank
    result = None
    try:
        if _remaining() > 15:
            result = _try_eliminate(U26, V26, W26)
    except Exception:
        result = None

    if result is not None:
        U, V, W = result
        # snap to half-integers (entries are already integers, but be safe)
        U = _snap_half_int(U)
        V = _snap_half_int(V)
        W = _snap_half_int(W)
        if _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to the verified rank-26 base
    U = _snap_half_int(U26)
    V = _snap_half_int(V26)
    W = _snap_half_int(W26)
    if _verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Last resort: schoolbook rank 27
    U27, V27, W27 = _build_rank27()
    return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
