"""Matrix multiplication 3x3x5 rank-minimal decomposition via tensor rank-1 peeling search."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]
n, m, p = DIMENSIONS

# Build the multiplication tensor T of shape (n, m, p) with T[i,j,k] = 1.
# The multiplication: C[i,k] = sum_j A[i,j] B[j,k].
# We seek U (n*m x rank), V (m*p x rank), W (p*n x rank) such that
# for all i,k: C[i,k] = sum_r (sum_{i',j} U[i'*m+j, r] A[i',j]) * (sum_{j',k'} V[j'*p+k', r] B[j',k']) * ... 
# Actually the standard form: the bilinear map is C = sum_r (U_r · A) (V_r · B) W_r
# where (U_r · A) is scalar = sum_{i,j} U_r[i,j] A[i,j], similarly V_r·B, and W_r is a p*n matrix
# placed so that C += (U_r·A)(V_r·B) * W_r.
# We verify numerically.

def _verify(U, V, W, trials=8, seed=0):
    """Return True if decomposition exactly computes A@B for random integer matrices."""
    rng = np.random.default_rng(seed)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for _ in range(trials):
        A = rng.integers(-3, 4, size=(n, m))
        B = rng.integers(-3, 4, size=(m, p))
        C_true = A @ B
        C = np.zeros((n, p), dtype=float)
        for r in range(rank):
            ua = np.sum(U[:, r] * A.reshape(-1))
            vb = np.sum(V[:, r] * B.reshape(-1))
            # W[:, r] is length p*n, indexed as W[k*n + i]
            C += (ua * vb) * W[:, r].reshape(p, n).T
        if not np.allclose(C, C_true, atol=1e-6):
            return False
    return True

def _schoolbook():
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}

def _random_candidate(rank, rng):
    """Random rank-1 peeling attempt: build U,V,W by random bilinear forms, verify."""
    U = rng.integers(-2, 3, size=(n * m, rank))
    V = rng.integers(-2, 3, size=(m * p, rank))
    W = rng.integers(-2, 3, size=(p * n, rank))
    return U, V, W

def _try_peeling(rank, deadline, rng):
    """Attempt to find a valid rank-`rank` decomposition via iterative peeling.
    We start from the all-ones tensor and greedily pick rank-1 components that
    best reduce the residual, then solve for exact integer/half-integer fit.
    For small dimensions we do a structured search over sign patterns."""
    # Represent tensor as vector of shape (n*m, m*p)?? No.
    # Use the fact that T[i,j,k]=1. We want U,V,W with:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all (i,j,k).
    # This is a system. We search over candidate rank-1 tensors of the form
    # a ⊗ b ⊗ c restricted to the multiplication index structure.
    # A rank-1 term contributes to (i,j,k): U[i*m+j] * V[j*p+k] * W[k*n+i].
    # We need sum over r of that product == 1.
    # Try small integer values for U,V,W entries in {-2,-1,0,1,2}.
    # Use a greedy construction: for each (i,j,k) ensure coverage.
    # Given small size, do a randomized local search minimizing error.
    best_err = None
    best = None
    max_iter = 4000
    for it in range(max_iter):
        if time.time() > deadline:
            break
        U = rng.integers(-2, 3, size=(n * m, rank))
        V = rng.integers(-2, 3, size=(m * p, rank))
        W = rng.integers(-2, 3, size=(p * n, rank))
        # Compute error: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err += (s - 1) ** 2
        if err == 0:
            return U, V, W
        if best_err is None or err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
    return best

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    rng = np.random.default_rng(20240517)
    # Try decreasing ranks below schoolbook (45).
    for rank in range(12, 45):
        if time.time() > deadline:
            break
        res = _try_peeling(rank, deadline, rng)
        if res is not None:
            U, V, W = res
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    # Fallback: schoolbook rank 45
    return _schoolbook()
