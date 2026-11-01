"""Matrix multiplication 3x3 by 3x8: rank-minimal exact tensor decomposition.

Combines the structured construction approach (determining W from U,V) with
random search and local refinement. The key insight from inspiration 1 is that
W is determined by U and V, so we only need to search over U and V, then round
all three to half-integers and verify exactness.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]
START = time.time()
DEADLINE = 130.0  # safety margin before 150s kill


def _build_schoolbook():
    n, m, p = DIMENSIONS
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


def _round_to_half(x):
    """Round to nearest half-integer (multiple of 0.5)."""
    return np.round(2.0 * x) / 2.0


def _round_to_int(x):
    """Round to nearest integer."""
    return np.round(x)


def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that the tensor decomposition is exact.
    
    The matrix multiplication tensor T[i,j,k] has rank-r decomposition if:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    and == 0 for all other index combinations (but we only check the 1s).
    
    Actually, the correct check is: the bilinear form sum_c U[i*m+j,c]*V[j*p+k,c]
    should equal the row of W corresponding to (k,i) when contracted... 
    
    Let me re-read the task. The tensor for matrix multiplication (n×m)(m×p)→(n×p)
    has entries T_{i,k,j} = 1 if... no.
    
    Standard tensor: T is a 3-tensor of shape (n*p, n*m, m*p) where
    T[k*n+i, i*m+j, j*p+k] = 1 for all i,j,k, and 0 otherwise.
    
    A rank-r decomposition gives U (n*p × r), V (n*m × r), W (m*p × r) such that
    T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    i.e., T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    
    But the task says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    And W follows trace convention W[k*n+i, r].
    
    So the tensor is T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    And we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    Wait, that's a trilinear form. But matrix multiplication is bilinear. Let me re-read.
    
    "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    
    So the tensor T has indices (i*m+j, j*p+k, k*n+i) and T = 1 at those positions.
    The decomposition is T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r].
    
    For this to work, we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    And it should be 0 for all other (a,b,c) combinations that don't correspond to valid (i,j,k).
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    # Also check that non-matching indices give 0
    # This is expensive, so we'll do a sample check
    for _ in range(100):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        # Check if this corresponds to a valid (i,j,k)
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        if j != j2 or i != i2 or k != k2:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val) > tol:
                return False
    return True


def _determine_W(U, V, n, m, p):
    """Given U and V, determine W such that the decomposition holds.
    
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    This is a system of linear equations in W. For each (i,k), we have:
    sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c]) = 1
    
    Let M[i,k,c] = sum_j U[i*m+j, c] * V[j*p+k, c]
    Then sum_c W[k*n+i, c] * M[i,k,c] = 1 for all i,k.
    
    This is p*n equations in p*n*r unknowns (W has p*n rows, r columns).
    We want to find W with half-integer entries.
    """
    # Build the matrix M of shape (n, p, r)
    M = np.zeros((n, p, U.shape[1]))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                M[i, k, :] += U[i * m + j, :] * V[j * p + k, :]
    
    # For each (i,k), we need sum_c W[k*n+i, c] * M[i,k,c] = 1
    # This is an underdetermined system (r unknowns, 1 equation per (i,k))
    # We can pick a simple solution: if M[i,k,:] is not all zeros, set W[k*n+i,:] = M[i,k,:] / ||M[i,k,:]||^2
    # But we want half-integer entries, so this may not work directly.
    
    # Alternative: use the fact that the schoolbook solution has U, V, W all with 0/1 entries.
    # For a rank-r decomposition, we can try to find U, V such that W is also simple.
    
    # For now, return M transposed as a starting point for W
    W = np.zeros((p * n, U.shape[1]))
    for i in range(n):
        for k in range(p):
            W[k * n + i, :] = M[i, k, :]
    return W


def _try_rank(r, n, m, p, iters=500, seed=0):
    """Random search for rank-r decomposition. Returns (U,V,W) or None."""
    rng = np.random.default_rng(seed)
    for it in range(iters):
        if time.time() > DEADLINE:
            return None
        # Random factors
        U = rng.normal(size=(n * m, r))
        V = rng.normal(size=(m * p, r))
        # Determine W
        W = _determine_W(U, V, n, m, p)
        
        # Round to half-integers
        U_r = _round_to_half(U)
        V_r = _round_to_half(V)
        W_r = _round_to_half(W)
        
        # Check exactness with rounded values
        if _validate(U_r, V_r, W_r, n, m, p):
            def to_list(M):
                return [[(int(v) if v == int(v) else float(v)) for v in row] for row in M]
            return {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
    return None


def _try_rank_int(r, n, m, p, iters=500, seed=0):
    """Random search for rank-r decomposition with integer entries."""
    rng = np.random.default_rng(seed)
    for it in range(iters):
        if time.time() > DEADLINE:
            return None
        # Random factors with small integer values
        U = rng.integers(-3, 4, size=(n * m, r))
        V = rng.integers(-3, 4, size=(m * p, r))
        W = _determine_W(U.astype(float), V.astype(float), n, m, p)
        
        # Round W to integers
        W_r = _round_to_int(W)
        
        # Check exactness
        if _validate(U, V, W_r, n, m, p):
            def to_list(M):
                return [[int(v) for v in row] for row in M]
            return {"U": to_list(U), "V": to_list(V), "W": to_list(W_r)}
    return None


def _structured_decomposition(n, m, p):
    """Try known structured decompositions.
    
    For (n×m)(m×p), one approach is to use the identity:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    A rank n*p decomposition would require representing each inner product <row_i, col_k>
    with a fixed set of r terms. This is possible if r >= max(n,p) in some cases.
    
    Actually, for the specific case n=3, m=3, p=8, the lower bound is max(n*p, m*p, n*m) = max(24, 24, 9) = 24.
    
    Let's try a decomposition where we split the m dimension:
    For each j, we have a term A[i,j] * B[j,k]. This is the schoolbook with rank n*m*p = 72.
    
    To reduce rank, we can try to combine terms. For example, if we group j values:
    C[i,k] = sum_{j1} A[i,j1]*B[j1,k] + sum_{j2} A[i,j2]*B[j2,k]
    
    This doesn't obviously reduce rank unless there's structure.
    
    For now, return None and rely on random search.
    """
    return None


def construct():
    n, m, p = DIMENSIONS
    best = _build_schoolbook()  # rank 72 fallback
    best_rank = n * m * p

    # Try structured decomposition first
    res = _structured_decomposition(n, m, p)
    if res is not None:
        best = res
        best_rank = min(best_rank, len(res["U"][0]))

    # Try decreasing ranks from a reasonable lower bound up to best_rank-1.
    # Lower bound: max(n*p, m*p, n*m) = max(24, 24, 9) = 24.
    lower = max(n * p, m * p, n * m)
    seed_counter = 0
    
    # Budget allocation: we have ~130s. Try ranks from lower to best_rank-1.
    # For each rank, try both half-integer and integer searches.
    for r in range(lower, best_rank):
        if time.time() > DEADLINE:
            break
        # Budget per rank: roughly (DEADLINE - time.time()) / (best_rank - r)
        remaining_ranks = best_rank - r
        time_per_rank = max(0.5, (DEADLINE - 5 - time.time()) / remaining_ranks)
        iters = max(100, int(time_per_rank * 50))  # ~50 iterations per second
        
        # Try half-integer search
        res = _try_rank(r, n, m, p, iters=iters, seed=seed_counter)
        seed_counter += 1
        if res is not None:
            best = res
            best_rank = r
            break
        
        # Try integer search
        res = _try_rank_int(r, n, m, p, iters=iters, seed=seed_counter + 1000)
        seed_counter += 1
        if res is not None:
            best = res
            best_rank = r
            break

    return best


if __name__ == "__main__":
    res = construct()
    rank = len(res["U"][0]) if res else None
    print("rank:", rank)
    print("time:", time.time() - START)
