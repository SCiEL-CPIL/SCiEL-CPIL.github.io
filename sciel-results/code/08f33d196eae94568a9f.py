"""Matrix multiplication 4x5x5 — attempt to find rank < 100 decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 5]
DEADLINE = 140.0  # seconds; evaluator kills at 150s


def _schoolbook():
    n, m, p = DIMENSIONS
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-9):
    """Check exact decomposition. Returns True if valid."""
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    R = Ua.shape[1]
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We can check this by building the tensor via einsum
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
    Ur = Ua.reshape(n, m, R)
    Vr = Va.reshape(m, p, R)
    Wr = Wa.reshape(p, n, R)
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # Use einsum: 'jcr,jkr,ikr->ijk'  -- wait, indices need care
    # Ur[i,j,c], Vr[j,k,c], Wr[k,i,c]
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr, optimize=True)
    # The matrix multiplication tensor has T[i,j,k] = 1 if we use the standard basis
    # Actually for matrix mult (A_{n x m} * B_{m x p})_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T[i,j,k] = delta_{i,i} * delta_{k,k} ... no.
    # The tensor for matrix mult is: T[i,j,k] = 1 for all valid (i,j,k)
    # because C_{i,k} = sum_j A_{i,j} B_{j,k}, so the structure tensor has a 1 at every (i,j,k).
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=tol)


def _try_rank99_random():
    """Try to find a rank-99 decomposition by starting from schoolbook and zeroing one term,
    then trying to fix the residual with small perturbations of other terms."""
    n, m, p = DIMENSIONS
    R0 = n * m * p  # 100
    # Start from schoolbook
    U = np.zeros((n * m, R0))
    V = np.zeros((m * p, R0))
    W = np.zeros((p * n, R0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    # Try zeroing each term and see if we can fix it
    rng = np.random.default_rng(42)
    for zero_idx in range(R0):
        if time.time() > DEADLINE:
            return None
        # Set term zero_idx to zero
        U[:, zero_idx] = 0
        V[:, zero_idx] = 0
        W[:, zero_idx] = 0

        # The missing term is (i0, j0, k0) where zero_idx = i0*m*p + j0*p + k0
        i0 = zero_idx // (m * p)
        j0 = (zero_idx % (m * p)) // p
        k0 = zero_idx % p

        # We need sum_c U[i0*m+j0,c] * V[j0*p+k0,c] * W[k0*n+i0,c] = 1
        # but currently it's 0. We can try to add 1 to one of the existing terms
        # that shares the same (i0, j0) pair in U and (j0, k0) in V and (k0, i0) in W.
        # Actually, any term c where U[i0*m+j0,c] != 0, V[j0*p+k0,c] != 0, W[k0*n+i0,c] != 0
        # would contribute to T[i0,j0,k0].
        # In schoolbook, each term c corresponds to a unique (i,j,k).
        # Term c has U[i*m+j,c]=1 only for its specific (i,j), similarly for V and W.
        # So no other term shares all three indices.
        # This means we can't simply fix by adjusting one term.
        # Instead, we need to modify multiple terms.
        # Let's try: pick a term c1 with same i0, j0 (different k), and term c2 with same j0, k0 (different i).
        # This is getting complex. Let's try a different approach.
        break

    # Alternative: use a rank-99 decomposition found in literature or derived from Strassen-like ideas.
    # For 4x5x5, the schoolbook rank is 100.
    # It's known that for small matrices, the rank can sometimes be reduced.
    # Let's try a simple approach: can we merge two terms?
    # Term (i,j,k1) and term (i,j,k2) share U[i*m+j, c1] and U[i*m+j, c2].
    # If we set U[i*m+j, c1] = 2, U[i*m+j, c2] = -1, and adjust V, W accordingly...
    # This is essentially trying to find a linear dependency.

    # Let's try: for a fixed (i,j), can we reduce the number of k terms?
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For fixed i,j, we have p=5 values of k, each needing to be 1.
    # The U vector for (i,j) has 100 entries (one per term).
    # V and W vectors for (j,k) and (k,i) also have 100 entries.
    # This is a 5-dimensional constraint per (i,j) pair.
    # With 20 (i,j) pairs, we have 100 constraints total (one per (i,j,k)).

    # Let's try a random search: perturb a few entries and check.
    for attempt in range(200):
        if time.time() > DEADLINE:
            return None
        U2 = U.copy()
        V2 = V.copy()
        W2 = W.copy()
        # Zero out one term
        U2[:, zero_idx] = 0
        V2[:, zero_idx] = 0
        W2[:, zero_idx] = 0
        # Perturb a random subset of other terms
        others = [c for c in range(R0) if c != zero_idx]
        perturb_idx = rng.choice(others, size=min(5, len(others)), replace=False)
        for c in perturb_idx:
            delta = rng.uniform(-0.5, 0.5)
            # Modify U, V, W for term c
            # We need to find i,j,k for term c
            ci = c // (m * p)
            cj = (c % (m * p)) // p
            ck = c % p
            U2[ci * m + cj, c] += delta
            V2[cj * p + ck, c] += delta
            W2[ck * n + ci, c] += delta

        if _verify(U2.tolist(), V2.tolist(), W2.tolist(), tol=1e-6):
            # Found a valid decomposition! Convert to integers/half-integers if possible
            return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    return None


def construct():
    start = time.time()

    # First, try to find a rank-99 decomposition
    result = _try_rank99_random()
    if result is not None and time.time() < DEADLINE:
        return result

    # Fall back to schoolbook
    return _schoolbook()


# EVOLVE-BLOCK-END
