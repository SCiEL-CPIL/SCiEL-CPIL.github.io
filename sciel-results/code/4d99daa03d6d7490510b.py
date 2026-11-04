"""Suite task: rank-minimal exact tensor decomposition for 2x3 x 3x9 matrix product.

Strategy:
  - Hard deadline guard (wall clock) so the program always returns within budget.
  - Randomized + local-refinement search over the finite alphabet {0, +/-1/2, +/-1}
    for a low-rank exact decomposition of the matrix-multiplication tensor.
  - Exactness is checked by verifying the tensor identity on a rational basis of
    random (and structured) inputs using exact integer arithmetic (scaled by 2).
  - If no verified lower-rank solution is found in time, fall back to the
    guaranteed schoolbook decomposition (rank = n*m*p = 54), so the score is
    at worst the seed (1.0) and never 0.

The interface and return value are preserved: construct() -> {'U','V','W'}.
"""

import time
import random
import itertools
import numpy as np

DIMENSIONS = [2, 3, 9]
N, M, P = DIMENSIONS

ALPHABET = [0, 1, -1, 0.5, -0.5]

# Scale factor so that all entries become integers when multiplied.
SCALE = 2


def _to_int_matrix(A):
    """Convert a float matrix with half-integer entries to exact integers."""
    return np.round(np.asarray(A, dtype=float) * SCALE).astype(object)


def _matmul_tensor_check(U, V, W, n, m, p):
    """
    Exact verification of the decomposition:
      For all A (n x m), B (m x p):  (A B)[i,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    checked exactly on a set of basis inputs.

    We verify the bilinear identity: for each (i,j) in A and (j',k) in B with j==j',
    the coefficient of A[i,j] * B[j,k] in the product must equal 1, and all other
    cross-terms (i2,j2) x (j2,k2) with the same output (i,k) but not matching must sum to 0.

    Exact algebraic check: define the tensor T[i,k,(i2,j2),(j2,k2)] = delta[i2,i] delta[k2,k] delta[j2, j2]
    (i.e., the coefficient of A[i2,j2]*B[j2,k2] in output[i,k] is 1 iff i2==i and k2==k, else 0;
    j2 is shared). We require:
      sum_r U[i2,j2,r] * V[j2,k2,r] * W[k,n? ...]
    with W indexed by (k, i) per the trace convention W[k*n+i, r].

    So the contribution to output[i,k] from term (i2,j2) in A and (j2',k2) in B (j2==j2' required
    for the product to be a valid bilinear term) is:
      sum_r U[i2*m + j2, r] * V[j2*p + k2, r] * W[k2*n + i2, r]  ... but this mixes i2,k2 with i,k.
    Actually the output index is fixed (i,k). The decomposition says output[i,k] = sum_r W[k*n+i, r] * (something).
    Let me use the standard form: output[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_j' V[j'*p+k, r] B[j',k]).
    Expanding: output[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j' V[j'*p+k, r] B[j',k]).
    So the coefficient of A[i2,j] * B[j',k2] in output[i,k] is
      sum_r W[k*n+i, r] * U[i*m+j, r] * V[j'*p+k2, r]   with the constraint that this contributes only
      to output[i,k] (fixed i,k), and A's row is i, B's column is k2. Wait, A's index is (i2 = i) and B's is (j' = ?).
    For a bilinear term A[i2, j] B[j, k2] to appear in output[i, k] we need i2 == i and k2 == k.
    Coefficient of A[i, j] B[j, k] in output[i, k] must be 1.
    Coefficient of A[i, j] B[j, k'] (k' != k) in output[i, k] must be 0.
    Coefficient of A[i', j] B[j, k] (i' != i) in output[i, k] must be 0.
    Coefficient of A[i', j] B[j, k'] in output[i, k] (i'!=i, k'!=k) must be 0.

    General: coefficient of A[i2, j] B[j2, k2] in output[i, k] (valid only if j == j2) is
      C = sum_r W[k*n+i, r] * U[i2*m + j, r] * V[j*p + k2, r]
    (j == j2). This must equal delta[i2, i] * delta[k2, k].
    """
    Ui = _to_int_matrix(U)
    Vi = _to_int_matrix(V)
    Wi = _to_int_matrix(W)
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            for i2 in range(n):
                for k2 in range(p):
                    for j in range(m):
                        # coefficient of A[i2,j] B[j,k2] in output[i,k]
                        C = 0
                        for r in range(rank):
                            C += int(Wi[k * n + i, r]) * int(Ui[i2 * m + j, r]) * int(Vi[j * p + k2, r])
                        expected = (1 if (i2 == i and k2 == k) else 0)
                        if C != expected:
                            return False
    return True


def _schoolbook():
    n, m, p = N, M, P
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _random_candidate(rank, rng):
    n, m, p = N, M, P
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for arr in (U, V, W):
        for row in range(arr.shape[0]):
            for col in range(arr.shape[1]):
                # sparse: each cell is 0 with prob 0.5
                if rng.random() < 0.5:
                    arr[row, col] = ALPHABET[rng.randrange(len(ALPHABET))]
    return U, V, W


def _local_refine(U, V, W, rank, rng, deadline, best_rank_holder):
    """Greedy local search: flip a random cell in a random matrix to a random alphabet
    value if it does not make a sampled check worse; keep exactness via periodic full check.
    This is a heuristic; correctness is enforced by final exact check only.
    We return (U,V,W) unchanged if nothing improves; caller verifies exactness.
    """
    n, m, p = N, M, P
    cur = (U.copy(), V.copy(), W.copy())
    # We do NOT track an objective that guarantees exactness; instead we try random
    # mutations and only accept if a full exact check passes (expensive). To stay
    # within budget we do a bounded number of full checks.
    attempts = 0
    max_attempts = 40
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        # pick a matrix and a cell
        which = rng.randrange(3)
        arr = cur[which]
        row = rng.randrange(arr.shape[0])
        col = rng.randrange(arr.shape[1])
        newval = ALPHABET[rng.randrange(len(ALPHABET))]
        old = arr[row, col]
        arr[row, col] = newval
        # quick exact check (full, but rank is smallish)
        if _matmul_tensor_check(cur[0], cur[1], cur[2], n, m, p):
            return cur[0], cur[1], cur[2]
        arr[row, col] = old
    return cur[0], cur[1], cur[2]


def construct():
    start = time.time()
    # Leave a safety margin so we always return well before 120s.
    deadline = start + 110.0

    rng = random.Random(20240517)

    # First, try a handful of low ranks with random init + local refine.
    # Theoretical lower bound is at least max(n*m, m*p, p*n) = max(6,27,18)=27.
    # We try ranks from a small set up to a cap, and also random ranks.
    best = None
    best_rank = N * M * P  # 54

    ranks_to_try = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 40, 45, 50]
    # also a couple of random ranks below 54
    for _ in range(8):
        r = rng.randint(27, 53)
        if r not in ranks_to_try:
            ranks_to_try.append(r)

    for rank in ranks_to_try:
        if time.time() > deadline - 5.0:
            break
        if rank >= best_rank:
            continue
        # a few random restarts at this rank
        restarts = 6 if rank <= 30 else 4
        for _ in range(restarts):
            if time.time() > deadline - 5.0:
                break
            U, V, W = _random_candidate(rank, rng)
            # quick exact check on the random init
            if _matmul_tensor_check(U, V, W, N, M, P):
                best = (U, V, W)
                best_rank = rank
                break
            # local refine
            U2, V2, W2 = _local_refine(U, V, W, rank, rng, deadline, best_rank)
            if _matmul_tensor_check(U2, V2, W2, N, M, P):
                best = (U2, V2, W2)
                best_rank = rank
                break
        if best is not None and best_rank == rank:
            # found at this rank; try to go lower by continuing loop (loop will hit lower ranks later? no, ranks_to_try sorted asc). break to try even lower.
            pass

    if best is not None:
        U, V, W = best
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: guaranteed schoolbook decomposition (rank 54). Always valid.
    return _schoolbook()


# EVOLVE-BLOCK-END
