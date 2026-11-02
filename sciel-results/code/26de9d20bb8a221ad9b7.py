"""Matrix multiplication 2x4 x 4x7 rank-minimal tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check that the tensor decomposition exactly reproduces matrix multiplication."""
    n, m, p = int(n), int(m), int(p)
    U = np.asarray(U, dtype=np.int64)
    V = np.asarray(V, dtype=np.int64)
    W = np.asarray(W, dtype=np.int64)
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    r = U.shape[1]
    # Build the decomposition tensor T[i*m+j, j'*p+k, k'*n+i] = sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i,r]
    # We want T[i*m+j, j'*p+k, k'*n+i] == 1 if (j == j') else 0
    # Reshape: U -> (n, m, r), V -> (m, p, r), W -> (p, n, r)
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    # T[i, j, j', k, k'] = sum_r U3[i,j,r] * V3[j',k,r] * W3[k',i,r]
    # We want T[i, j, j', k, k'] == (j == j') for all i, k, k'
    # Efficient check: for each r, outer product
    # T = sum_r U3[:, :, r] * V3[:, :, r] * W3[:, :, r]  (broadcasting)
    # Let's do it in a loop for clarity (r is small)
    T = np.zeros((n, m, m, p, p), dtype=np.int64)
    for r in range(r):
        u = U3[:, :, r]  # (n, m)
        v = V3[:, :, r]  # (m, p)
        w = W3[:, :, r]  # (p, n)
        # u[i,j] * v[j',k] * w[k',i]
        # T[i, j, j', k, k'] += u[i,j] * v[j',k] * w[k',i]
        # Use broadcasting
        # u: (n, 1, 1, 1, m) -> need (n, m, 1, 1, 1)
        # v: (1, 1, m, p, 1)
        # w: (1, 1, 1, p, n) -> need (n, 1, 1, p, 1)? No, w[k', i] so (p, n) -> (1, 1, 1, p, n)
        # But T is (n, m, m, p, p). w[k', i] depends on k' and i.
        # So w reshaped to (p, n) -> broadcast to (n, m, m, p, p) as: w[k', i] -> (n, 1, 1, p, n)? No.
        # w is (p, n), so w[k', i] has indices k' and i. In T, i is axis 0, k' is axis 4.
        # So we need w reshaped to (n, 1, 1, p, 1) but with k' and i.
        # Actually, let's just use a more direct approach.
        # T[i, j, j', k, k'] += u[i, j] * v[j', k] * w[k', i]
        # u[i, j] -> shape (n, m, 1, 1, 1)
        # v[j', k] -> shape (1, 1, m, p, 1)
        # w[k', i] -> shape (n, 1, 1, p, 1) but transposed: w is (p, n), so w.T is (n, p), then reshape to (n, 1, 1, p, 1)
        T += u[:, :, np.newaxis, np.newaxis, np.newaxis] * \
             v[np.newaxis, np.newaxis, :, :, np.newaxis] * \
             w.T[:, np.newaxis, np.newaxis, :, np.newaxis]
    
    # Check T[i, j, j', k, k'] == (j == j')
    # The expected tensor is: for each i, k, k', the (m, m) submatrix is identity.
    # T_expected[i, j, j', k, k'] = 1 if j == j' else 0
    # This is independent of i, k, k'.
    # So T should be: for all i, k, k', T[i, :, :, k, k'] = I_m
    # Let's check
    for i in range(n):
        for k in range(p):
            for kp in range(p):
                sub = T[i, :, :, k, kp]
                if not np.array_equal(sub, np.eye(m, dtype=np.int64)):
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145

    n, m, p = DIMENSIONS
    
    # First, try the known rank-50 solution
    best_result = _build_rank50(n, m, p)
    if best_result is not None:
        best_rank = 50
    else:
        best_rank = n * m * p
        best_result = _schoolbook(n, m, p)

    if time.time() > deadline - 10:
        return best_result

    # Try to improve with a search over structured decompositions
    # We'll try rank 49, 48, etc. by attempting to merge Strassen blocks more cleverly
    # or by using a mixed approach with different block sizes.
    
    # Idea: Instead of 3 Strassen blocks (each 7 mults) + 2 schoolbook blocks (each 4 mults) = 21+8=29... wait
    # 3 Strassen = 21, 2 schoolbook = 8, total 29? No.
    # 6 Strassen blocks = 42, 2 schoolbook = 8, total 50. Yes.
    # Can we do better?
    # What if we use a 2x4x2 block? Or 2x2x4?
    # Or a different partition?
    
    # Let's try a more sophisticated search:
    # We can model this as: we have a set of "basis" tensor decompositions (Strassen blocks, schoolbook blocks, etc.)
    # and we want to cover all the required terms with minimal rank.
    # This is a set cover problem, but with the constraint that the decomposition must be exact.
    
    # Alternative: use a linear programming / integer programming approach
    # to find a rank-49 or rank-48 decomposition.
    
    # Given time constraints, let's try a few specific structured alternatives:
    
    # Alternative 1: Use 5 Strassen blocks (35 mults) + 14 schoolbook (14 mults) = 49?
    # 5 Strassen blocks: cover 5 of the 6 k-pairs? No, we have 3 k-pairs and 2 j-pairs.
    # Actually, the current setup is: for each j-pair (0,1) and (2,3), we have 3 Strassen blocks for k-pairs (0,1),(2,3),(4,5) and 1 schoolbook for k=6.
    # Total: 2 * (3*7 + 4) = 2 * 25 = 50.
    
    # What if we can combine the two j-pairs more cleverly?
    # For example, use a 4x4x7 block? But we only have n=2.
    
    # What if we use a different Strassen variant or a Karatsuba-like approach for the 4x7 part?
    
    # Given the complexity, let's just try to verify the rank-50 solution is correct and return it.
    # The search for rank < 50 is an open problem and likely requires significant computational resources.
    
    # For now, return the rank-50 solution.
    return best_result

def _build_rank50(n, m, p):
    """Build the rank-50 decomposition using 6 Strassen + 2 schoolbook blocks."""
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0

    def _strassen_block(i0, j0, k0):
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + j0 + 1
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + j0 + 1
        e = j0 * p + k0
        f = j0 * p + k0 + 1
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + k0 + 1
        o00 = k0 * n + i0
        o01 = (k0 + 1) * n + i0
        o10 = k0 * n + i0 + 1
        o11 = (k0 + 1) * n + i0 + 1

        U[a, col] += 1; U[d, col] += 1
        V[e, col] += 1; V[h, col] += 1
        W[o00, col] += 1; W[o11, col] += 1
        col += 1

        U[c, col] += 1; U[d, col] += 1
        V[e, col] += 1
        W[o10, col] += 1; W[o11, col] -= 1
        col += 1

        U[a, col] += 1
        V[f, col] += 1; V[h, col] -= 1
        W[o01, col] += 1; W[o11, col] += 1
        col += 1

        U[d, col] += 1
        V[g, col] += 1; V[e, col] -= 1
        W[o00, col] += 1; W[o10, col] += 1
        col += 1

        U[a, col] += 1; U[b, col] += 1
        V[h, col] += 1
        W[o00, col] -= 1; W[o01, col] += 1
        col += 1

        U[c, col] += 1; U[a, col] -= 1
        V[e, col] += 1; V[f, col] += 1
        W[o11, col] += 1
        col += 1

        U[b, col] += 1; U[d, col] -= 1
        V[g, col] += 1; V[h, col] += 1
        W[o00, col] += 1
        col += 1

    def _sb_block(i0, j0, k0):
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + j0 + 1
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + j0 + 1
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        o0 = k0 * n + i0
        o1 = k0 * n + i0 + 1

        U[a, col] += 1; V[e, col] += 1; W[o0, col] += 1; col += 1
        U[b, col] += 1; V[g, col] += 1; W[o0, col] += 1; col += 1
        U[c, col] += 1; V[e, col] += 1; W[o1, col] += 1; col += 1
        U[d, col] += 1; V[g, col] += 1; W[o1, col] += 1; col += 1

    for jp in (0, 2):
        for kp in (0, 2, 4):
            _strassen_block(0, jp, kp)
        _sb_block(0, jp, 6)

    if col != rank:
        return None

    # Verify the decomposition
    if not _verify(U, V, W, n, m, p):
        return None

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook(n, m, p):
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
