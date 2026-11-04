"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import sys

def _wftt_decomp(n=4, m=4, p=5, L=5):
    """WFTT-style cyclic convolution embedding: build U,V,W of rank L*(p-1)=20.

    Indexing: U is (n*m, r), V is (m*p, r), W is (p*n, r).
    We map (a,c,d) with a in 0..3 (L-1=4 positions), c in 0..4 (5 positions), d in 0..3.
    term index = (c*(L-1) + a)*(p-1) + d  (arbitrary; 20 terms).
    """
    rank = L * (p - 1)  # 5*4 = 20
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Place (i,j,k) into term where c = (i+j) mod L, a = i, d = j.
                # This requires L >= m so a=i in [0,m-1] and the column index
                # for the cyclic term is well-defined.
                a = i
                d = j
                c_idx = (i + j) % L
                col = (c_idx * (L - 1) + a) * (p - 1) + d
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
    return {"U": U, "V": V, "W": W}


def _verify(decomp, n=4, m=4, p=5, iters=25, tol=1e-6):
    """Check the decomposition exactly on the tensor (via a random identity check).

    A tensor decomposition U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 iff (i,j,k) else 0
    means: for every r, the rank-1 term U[:,:,r] * V[:,:,r] * W[:,:,r] must be
    0 unless it corresponds to a valid (i,j,k), in which case the SUM over r
    equals the indicator of the tensor.

    Equivalent check: the bilinear form computed from U,V,W equals the bilinear
    form of matrix multiplication on random inputs, with W in trace convention.
    We instead check the *tensor directly*: for every r, define T_r(i,j,k) =
    U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]. Summing over r must equal 1 at the
    valid positions and 0 elsewhere. For (i,j,k), the tensor T is 1.
    """
    U = np.array(decomp["U"], dtype=float)
    V = np.array(decomp["V"], dtype=float)
    W = np.array(decomp["W"], dtype=float)
    rank = U.shape[1]
    # Build the tensor from decomposition: T(i,j,k) = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # Build the true tensor: T_true(i,j,k) = 1.
    for r in range(rank):
        pass  # (we do a single check via bilinear random test)
    # Random bilinear test: (A U) * (V B) then contract with W.
    # Actually the correct check: for a bilinear algorithm with U (inputs of left),
    # V (inputs of right), W (combination), the output[i,k] = sum_r (sum_j A[i,j]*U[i*m+j,r])
    #                                                        * (sum_j V[j*p+k,r]*B[j,k])
    #                                                       * W[k*n+i, r]
    # So W indexes (k, i) -> trace convention W[k*n + i].
    for _ in range(iters):
        A = np.random.randint(-99, 100, (n, m))
        B = np.random.randint(-99, 100, (m, p))
        expected = A @ B
        # Compute bilinear
        # X[i, r] = sum_j A[i,j] * U[i*m + j, r]
        X = np.zeros((n, rank))
        Y = np.zeros((rank, p))
        for i in range(n):
            for j in range(m):
                X[i, :] += A[i, j] * U[i * m + j, :]
        for j in range(m):
            for k in range(p):
                Y[:, k] += B[j, k] * V[j * p + k, :]
        # Z[i, k] = sum_r X[i, r] * Y[r, k] * W[k*n + i, r]
        Z = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                Z[i, k] = np.dot(X[i, :], Y[:, k] * W[k * n + i, :])
        if np.max(np.abs(Z - expected)) > tol:
            return False
    return True


def _schoolbook(n=4, m=4, p=5):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def construct():
    t0 = time.time()
    DEADLINE = 120
    SAFETY = 10

    n, m, p = DIMENSIONS

    # Candidate 1: WFTT-derived rank 20 decomposition.
    try:
        cand = _wftt_decomp(n, m, p, L=m)  # L=5
        if time.time() - t0 < DEADLINE - SAFETY:
            if _verify(cand, n, m, p, iters=20):
                # Convert numpy -> lists of plain ints for stable serialization
                out = {
                    "U": [[int(x) for x in row] for row in np.array(cand["U"], dtype=int)],
                    "V": [[int(x) for x in row] for row in np.array(cand["V"], dtype=int)],
                    "W": [[int(x) for x in row] for row in np.array(cand["W"], dtype=int)],
                }
                return out
    except Exception:
        pass

    # Candidate 2: fallback schoolbook.
    return _schoolbook(n, m, p)


# EVOLVE-BLOCK-END
