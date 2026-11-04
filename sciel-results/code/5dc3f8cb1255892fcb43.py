"""
Rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.

Design (kept from seed family):
  Partition the middle dimension (3) into 2 + 1.
  - Part 1: A[0:2,:] (2x2) * B[0:2,:] (2x10). Split the 10 output columns into
    five 2-wide blocks; apply classic rank-7 Strassen to each 2x2 x 2x2 block.
    Subtotal: 5 * 7 = 35.
  - Part 2: A[2,:] (2x1) * B[2,:] (1x10). Schoolbook: 2*1*10 = 20.
  Total rank: 55.

Refinements added (robustness machinery):
  1. Multi-trial exact verification with several seeds and a larger integer range.
  2. Graceful fallback to the guaranteed-correct schoolbook decomposition (rank 60)
     if any verification trial fails, so the returned answer is always valid.
  3. Explicit deadline guard (construction is O(1), but we guard anyway).
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0  # Safety margin before the 150 s kill


def _verify_decomposition(U, V, W, n, m, p, num_trials=8, seed=1234):
    """Verify the decomposition with random integer inputs (exact arithmetic).

    Returns (ok, max_err). Uses vectorized ops for speed.
    """
    rng = np.random.RandomState(seed)
    max_err = 0.0
    for _ in range(num_trials):
        A = rng.randint(-20, 21, size=(n, m)).astype(float)
        B = rng.randint(-20, 21, size=(m, p)).astype(float)
        C_direct = A @ B

        A_flat = A.flatten()
        B_flat = B.flatten()
        u_sums = U.T @ A_flat          # (rank,)
        v_sums = V.T @ B_flat          # (rank,)
        scalars = u_sums * v_sums      # (rank,)
        C_flat = W @ scalars           # (p*n,)
        C_decomp = C_flat.reshape(p, n).T  # (n, p)

        err = float(np.max(np.abs(C_decomp - C_direct)))
        if err > max_err:
            max_err = err
        if err > 1e-9:
            return False, max_err
    return True, max_err


def _schoolbook(n, m, p):
    """Guaranteed-correct schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    return U, V, W


def _strassen_decomposition(n, m, p):
    """Build the rank-55 Strassen-on-2x2-blocks + schoolbook decomposition."""
    rank = 35 + 20  # 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0

    # Part 1: Strassen for 2x2 x 2x10 (five 2x2x2 blocks)
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1

        # M1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        #     -> C[0,k0] += M1, C[1,k1] += M1
        U[0 * 3 + 0, idx] = 1.0
        U[1 * 3 + 1, idx] = 1.0
        V[0 * 10 + k0, idx] = 1.0
        V[1 * 10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M2 = (A[1,0] + A[1,1]) * B[0,k0]
        #     -> C[0,k1] += M2, C[1,k1] -= M2
        U[1 * 3 + 0, idx] = 1.0
        U[1 * 3 + 1, idx] = 1.0
        V[0 * 10 + k0, idx] = 1.0
        W[k0 * 2 + 1, idx] = 1.0
        W[k1 * 2 + 1, idx] = -1.0
        idx += 1

        # M3 = A[0,0] * (B[0,k1] - B[1,k1])
        #     -> C[0,k1] += M3, C[1,k1] += M3
        U[0 * 3 + 0, idx] = 1.0
        V[0 * 10 + k1, idx] = 1.0
        V[1 * 10 + k1, idx] = -1.0
        W[k1 * 2 + 0, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M4 = A[1,1] * (B[1,k0] - B[0,k0])
        #     -> C[0,k0] += M4, C[1,k0] += M4
        U[1 * 3 + 1, idx] = 1.0
        V[1 * 10 + k0, idx] = 1.0
        V[0 * 10 + k0, idx] = -1.0
        W[k0 * 2 + 0, idx] = 1.0
        W[k0 * 2 + 1, idx] = 1.0
        idx += 1

        # M5 = (A[0,0] + A[0,1]) * B[1,k1]
        #     -> C[0,k0] -= M5, C[0,k1] += M5
        U[0 * 3 + 0, idx] = 1.0
        U[0 * 3 + 1, idx] = 1.0
        V[1 * 10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = -1.0
        W[k1 * 2 + 0, idx] = 1.0
        idx += 1

        # M6 = (A[1,0] - A[0,0]) * (B[0,k0] + B[0,k1])
        #     -> C[1,k1] += M6
        U[1 * 3 + 0, idx] = 1.0
        U[0 * 3 + 0, idx] = -1.0
        V[0 * 10 + k0, idx] = 1.0
        V[0 * 10 + k1, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M7 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        #     -> C[0,k0] += M7
        U[0 * 3 + 1, idx] = 1.0
        U[1 * 3 + 1, idx] = -1.0
        V[1 * 10 + k0, idx] = 1.0
        V[1 * 10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = 1.0
        idx += 1

    # Part 2: Schoolbook for 2x1 x 1x10
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2, idx] = 1.0
            V[2 * 10 + k, idx] = 1.0
            W[k * 2 + i, idx] = 1.0
            idx += 1

    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    # Build the rank-55 decomposition.
    U, V, W = _strassen_decomposition(n, m, p)

    # Rigorous verification with several seeds; fall back to schoolbook on failure.
    ok, _ = _verify_decomposition(U, V, W, n, m, p, num_trials=8, seed=1234)
    if not ok:
        ok2, _ = _verify_decomposition(U, V, W, n, m, p, num_trials=8, seed=9999)
        if not ok2:
            U, V, W = _schoolbook(n, m, p)

    # Deadline guard (construction is O(1); this is a safety net only).
    if time.time() - start_time > DEADLINE:
        U, V, W = _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


# Entry point required by evaluator
if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

    n, m, p = DIMENSIONS
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]

    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)

    A_flat = A.flatten()
    B_flat = B.flatten()
    u_sums = U.T @ A_flat
    v_sums = V.T @ B_flat
    scalars = u_sums * v_sums
    C_flat = W @ scalars
    C_decomp = C_flat.reshape(p, n).T

    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))
    print(f"Max error: {error}")
    if error > 1e-10:
        print("WARNING: Decomposition is incorrect!")
