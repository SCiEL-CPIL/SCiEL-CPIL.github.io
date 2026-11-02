"""Search for a rank-minimal exact tensor decomposition of 3x3 * 3x6 multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]
START = time.time()
DEADLINE = START + 140.0  # safety margin vs 150 s kill


def _time_left():
    return DEADLINE - time.time()


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _matmul_decomp(U, V, W):
    """Return (A,B,C) with A[i,j] = sum_c U[i,j,c]*V[i,j,c] and C[k,j] = sum_i A[i,k]*B[k,j]."""
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    W = np.asarray(W, dtype=object)
    A = np.zeros((n, m, p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A[i, j, k] = U[i * m + j, k] * V[j * p + k, k]
    C = np.zeros((p, n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                C[k, i] += A[i, j, k]
    return A, C


def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    W = np.asarray(W, dtype=object)
    A, C = _matmul_decomp(U, V, W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if A[i, j, k] != 1:
                    return False
    for i in range(n):
        for k in range(p):
            if C[k, i] != 0:
                return False
    return True


def _to_ints(U, V, W):
    """Scale to integer entries if possible; return (U,V,W) as int lists or None."""
    all_vals = []
    for M in (U, V, W):
        for row in M:
            for v in row:
                all_vals.append(v)
    # find common denominator
    def to_frac(x):
        if isinstance(x, (int, np.integer)):
            return (int(x), 1)
        if isinstance(x, float):
            # round to nearest half
            h = int(round(x * 2))
            return (h, 2)
        return (int(x), 1)
    nums, dens = [], []
    for v in all_vals:
        num, den = to_frac(v)
        nums.append(num)
        dens.append(den)
    # lcm of dens
    def lcm(a, b):
        from math import gcd
        return abs(a * b) // gcd(a, b) if a and b else max(a, b)
    L = 1
    for d in dens:
        L = lcm(L, d)
    U2 = [[int(round(float(v) * L)) for v in row] for row in U]
    V2 = [[int(round(float(v) * L)) for v in row] for row in V]
    W2 = [[int(round(float(v) * L)) for v in row] for row in W]
    # verify integer exactness
    if not _verify(U2, V2, W2):
        return None
    return U2, V2, W2


def _try_rank(rank, rng, max_iters=400):
    """Attempt to find an exact rank-`rank` decomposition via randomized linear algebra."""
    n, m, p = DIMENSIONS
    N = n * m
    M = m * p
    P = p * n
    best = None
    for it in range(max_iters):
        if _time_left() < 2.0:
            break
        U = rng.standard_normal((N, rank))
        V = rng.standard_normal((M, rank))
        # Build the tensor T as a matrix: rows indexed by (i,j,k), value 1.
        # Constraint: for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # and for each (i,k) with k != j*n... actually C[k,i]=sum_j A[i,j,k]=0
        # This is a system of linear equations in W given U,V.
        # For fixed U,V, we need W such that:
        #   For all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        #   For all i,k: sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 0
        # Note the second is implied by the first summed over j only if consistent.
        # Actually C[k,i] = sum_j A[i,j,k] and we need A[i,j,k]=1 for all, so C[k,i]=m.
        # Wait: C[k,i] = sum_j A[i,j,k] = sum_j 1 = m, not 0!
        # Re-examine: the output is C[k*n+i] = sum_j A[i,j,k]*B[j,k]... 
        # The convention is W[k*n+i, r]. The multiplication tensor T[i,j,k] = 1 means
        # C[i,j] (i-th row, j-th col of A*B) = sum_k A[i,k]*B[k,j].
        # With A (n x m) and B (m x p), C (n x p). C[i, j] = sum_k A[i,k]*B[k,j].
        # The factorization: C[i,j] = sum_r U[i*m+k... 
        # Standard: C[i,j] = sum_r (sum_k U[i*m+k, r]*A[i,k]) * (sum_k V[k*p+j, r]*B[k,j])
        # So U rows indexed by (i,k) for i in 0..n-1, k in 0..m-1 -> i*m+k
        # V rows indexed by (k,j) for k in 0..m-1, j in 0..p-1 -> k*p+j
        # W rows indexed by (i,j) for i in 0..n-1, j in 0..p-1 -> i*p+j
        # But the task says W[k*n+i, r] which is different indexing.
        # Let me re-read: "W[k*n+i,r]" and shapes (p*n, rank).
        # So W has p*n rows, indexed by k (0..p-1) and i (0..n-1): k*n+i.
        # This means the output index is (k, i) where k is column of C (0..p-1) and i is row (0..n-1).
        # So C[i, k] = sum_r ... and W[k*n+i, r] is the factor for output (i,k).
        # The tensor entry T[i, (j), k] for A[i,j]*B[j,k] contributes to C[i,k].
        # So the constraint is: for all i, j, k: sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = 1
        # And there are no other constraints (the "off-diagonal" C entries are exactly the sum over j).
        # So we just need the tensor equation to hold for all (i,j,k).
        # That's n*m*p = 54 equations in P*rank unknowns (W entries).
        # For fixed U, V, this is a linear system in W.
        # Number of unknowns: P * rank = 18 * rank.
        # Number of equations: 54.
        # We need rank such that 18*rank >= 54, i.e., rank >= 3.
        # Let's set up the linear system.
        # Equation (i,j,k): sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = 1
        # Let W_vec be a vector of length P*rank, W_vec[(k*n+i)*rank + r] = W[k*n+i, r]
        # Then equation (i,j,k) is: sum_r (U[i*m+j, r]*V[j*p+k, r]) * W_vec[(k*n+i)*rank + r] = 1
        # This is a linear system A_sys * W_vec = b where b is all ones (54 entries).
        # A_sys has 54 rows and P*rank columns.
        # For each (i,j,k) and each r: A_sys[(i,j,k), (k*n+i)*rank + r] = U[i*m+j, r]*V[j*p+k, r]
        # Let's build this.
        rows = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    rows.append((i, j, k))
        num_eq = len(rows)
        # Build A_sys as a numpy array of floats
        A_sys = np.zeros((num_eq, P * rank), dtype=float)
        for eq_idx, (i, j, k) in enumerate(rows):
            for r in range(rank):
                col = (k * n + i) * rank + r
                A_sys[eq_idx, col] = U[i * m + j, r] * V[j * p + k, r]
        b = np.ones(num_eq, dtype=float)
        # Solve least squares
        try:
            W_vec, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        # Check residual
        if len(residuals) > 0 and residuals[0] > 1e-8:
            continue
        # Reshape W_vec
        W = W_vec.reshape(P, rank)
        # Verify exactly
        U_list = [[float(U[i, r]) for r in range(rank)] for i in range(N)]
        V_list = [[float(V[i, r]) for r in range(rank)] for i in range(M)]
        W_list = [[float(W[i, r]) for r in range(rank)] for i in range(P)]
        if _verify(U_list, V_list, W_list):
            # Try to convert to integers
            int_result = _to_ints(U_list, V_list, W_list)
            if int_result is not None:
                return int_result
            else:
                # Accept float if exact
                return U_list, V_list, W_list
    return None


def construct():
    global DEADLINE
    DEADLINE = time.time() + 140.0
    # Start from schoolbook rank 54 and try to reduce
    schoolbook = _schoolbook()
    best_rank = 54
    best_result = schoolbook
    rng = random.Random(42)
    np_rng = np.random.default_rng(42)
    # Try ranks from 3 upward
    for rank in range(3, 54):
        if _time_left() < 5.0:
            break
        result = _try_rank(rank, np_rng, max_iters=200)
        if result is not None:
            U2, V2, W2 = result
            return {"U": U2, "V": V2, "W": W2}
    return best_result
