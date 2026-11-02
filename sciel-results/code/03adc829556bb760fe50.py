"""Tensor decomposition search for 3x3 by 3x7 matrix multiplication.

Strategy: exploit the additive structure of the output dimension (7 = 3 + 4).
The multiplication tensor is a direct sum over output-column blocks. We search
for a JOINT rank-minimal decomposition using a shared input factor U (size 9 x R)
and V (size 21 x R) with a sparse support pattern, and a block-structured output
factor W (size 21 x R). Randomized construction + verification + greedy support
pruning is used, with an explicit wall-clock deadline guard.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# Deadline guard: keep a safety margin before the 150s wall-clock kill.
DEADLINE = 135.0
_START = time.time()

# Output-column block split: 7 = 3 (core) + 4 (extension)
BLOCKS = [3, 4]


def _elapsed():
    return time.time() - _START


def _build_tensors():
    """Build the two direct-sum output blocks of the multiplication tensor."""
    n, m, p = DIMENSIONS
    # tensor B[b, i, j, k] = 1 iff output column (block b, local k) receives A[i,j]*X[j,k].
    tensors = []
    for start, size in zip([0, 3], BLOCKS):
        T = np.zeros((n, m, p), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(start, start + size):
                    T[i, j, k] = 1
        tensors.append(T)
    return tensors


def _verify(U, V, W, tensors, tol=1e-6):
    """Check that the decomposition reconstructs every block tensor exactly."""
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    for b, T in enumerate(tensors):
        start = 0 if b == 0 else BLOCKS[0]
        size = BLOCKS[b]
        # Reconstructed tensor: C[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        Wb = W[start * n:(start + size) * n, :]
        ok = True
        for i in range(n):
            for j in range(m):
                u = U[i * m + j, :]
                for k in range(start, start + size):
                    v = V[j * p + k, :]
                    w = Wb[k * n + i, :]
                    val = float(np.dot(np.dot(u, v), w))
                    if abs(val - T[i, j, k]) > tol:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            return False
    return True


def _greedy_prune(U, V, W, tensors, tol=1e-6):
    """Remove columns that do not affect the reconstruction (keep rank minimal)."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    keep = list(range(R))
    changed = True
    while changed and _elapsed() < DEADLINE:
        changed = False
        for r in list(keep):
            cand = [c for c in keep if c != r]
            Uc = U[:, cand]
            Vc = V[:, cand]
            Wc = W[:, cand]
            if _verify(Uc, Vc, Wc, tensors, tol):
                keep = cand
                changed = True
                break
    return U[:, keep], V[:, keep], W[:, keep]


def _random_decomposition(rank, tensors, seed):
    """Randomly sample a rank-`rank` decomposition and check validity."""
    n, m, p = DIMENSIONS
    rng = random.Random(seed)
    R = rank
    # Sparse-ish random integer factors (support pattern shared across blocks).
    def rand_factor(rows, sparsity):
        M = np.zeros((rows, R), dtype=float)
        for r in range(R):
            n_nonzero = max(1, int(round(rows * sparsity)))
            idx = rng.sample(range(rows), min(n_nonzero, rows))
            for i in idx:
                M[i, r] = float(rng.choice([-2, -1, 1, 2]))
        return M
    U = rand_factor(n * m, 0.4)
    V = rand_factor(m * p, 0.3)
    W = rand_factor(p * n, 0.4)
    return U, V, W


def _search_rank(rank, tensors, seed):
    """Try to find a valid decomposition of the given rank."""
    base = seed
    attempts = 0
    while _elapsed() < DEADLINE:
        U, V, W = _random_decomposition(rank, tensors, base + attempts)
        attempts += 1
        if _verify(U, V, W, tensors, tol=1e-6):
            U, V, W = _greedy_prune(U, V, W, tensors, tol=1e-6)
            return U, V, W
    return None


def construct():
    n, m, p = DIMENSIONS
    tensors = _build_tensors()
    schoolbook_rank = n * m * p

    # Best known valid decomposition (fallback to schoolbook).
    best = None

    # Lower bound: at least the max block rank contribution; search ascending.
    # Each block b needs rank at least max(n*m, m*block_size) in principle, but a
    # joint shared decomposition can be smaller. Search from a modest lower bound.
    lower = max(n * m, m * max(BLOCKS))  # = max(9, 12) = 12
    for rank in range(lower, schoolbook_rank + 1):
        if _elapsed() >= DEADLINE:
            break
        res = _search_rank(rank, tensors, seed=rank * 1000003)
        if res is not None:
            best = res
            break  # found minimal rank; stop

    if best is None:
        # Fallback: guaranteed schoolbook decomposition.
        R = schoolbook_rank
        U = np.zeros((n * m, R), dtype=float)
        V = np.zeros((m * p, R), dtype=float)
        W = np.zeros((p * n, R), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        best = (U, V, W)

    U, V, W = best
    # Convert to lists of integer/half-integer entries (round to nearest 0.5).
    def to_list(M):
        return [[round(float(x) * 2) / 2 for x in row] for row in np.asarray(M, dtype=float)]
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
