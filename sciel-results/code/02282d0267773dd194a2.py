import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]


def _build_schoolbook(n, m, p):
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


def _verify(U, V, W, n, m, p):
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > 1e-9:
                    return False
    T2 = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[i * m + j, j * p + k, k * n + i] = 0.0
    if np.max(np.abs(T2)) > 1e-9:
        return False
    return True


def _is_half_integer(x, tol=1e-9):
    return abs(x * 2 - round(x * 2)) < tol


def _build_strassen114(n, m, p):
    """The known rank-114 Strassen-based decomposition as starting point."""
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    U_sums = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1],
    ]
    V_sums = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1],
        [0, -1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1],
    ]
    W_sums = [
        [1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0],
        [1, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0],
    ]

    def add_444(col, a_base, b_base, c_base):
        for t in range(7):
            u, v, w = U_sums[t], V_sums[t], W_sums[t]
            for s in range(4):
                if u[s]:
                    U[a_base + s][col] += u[s]
                if v[s]:
                    V[b_base + s][col] += v[s]
                if w[s]:
                    W[c_base + s][col] += w[s]
            col += 1
        return col

    col = add_444(0, 0, 0, 0)
    col = add_444(col, 0, 4, 4)
    # 4x4x1 matrix-vector: rank 16
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[(j) * 1 + 0][col] = 1  # b_base+8, bc=1 -> index j*1+0
            W[0 * 4 + i][col] = 1    # c_base+8, pn: k*n+i = 0*4+i = i
            col += 1
    return {"U": U, "V": V, "W": W}


def _gauss_solve(A, b):
    """Exact rational Gauss elimination. A is list of lists of Fraction, b is list of Fraction.
    Returns list of Fraction solution or None if inconsistent."""
    n = len(A)
    m = len(A[0]) if n > 0 else 0
    # Augmented matrix
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]
    rank = 0
    for col in range(m):
        # Find pivot
        pivot = -1
        for r in range(rank, n):
            if aug[r][col] != 0:
                pivot = r
                break
        if pivot == -1:
            continue
        # Swap
        aug[rank], aug[pivot] = aug[pivot], aug[rank]
        # Normalize
        pv = aug[rank][col]
        for c in range(col, m + 1):
            aug[rank][c] /= pv
        # Eliminate
        for r in range(n):
            if r != rank and aug[r][col] != 0:
                factor = aug[r][col]
                for c in range(col, m + 1):
                    aug[r][c] -= factor * aug[rank][c]
        rank += 1
    # Check consistency
    for r in range(rank, n):
        if any(aug[r][c] != 0 for c in range(m)):
            return None
    # Extract solution (square system assumed)
    if rank != m:
        return None
    sol = [aug[i][m] for i in range(m)]
    return sol


def _sparsify(res, n, m, p, deadline):
    """Try to remove basis vectors one at a time via exact integer re-solve."""
    U = [row[:] for row in res["U"]]
    V = [row[:] for row in res["V"]]
    W = [row[:] for row in res["W"]]
    rank = len(U[0]) if U else 0

    # Build the linear system matrix
    # For each output (i,j,k) and each rank r, coefficient is U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    # We want sum_r coeff[r] * x[r] = 1 for (i,j,k) in support, = 0 otherwise
    # This is linear in x (the weights), but the coefficients depend on U,V,W structure.
    # Actually, the tensor is T[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] * 1
    # If we fix U,V,W structure (which entries are nonzero and their values), then T is linear in... 
    # Wait, no. The decomposition is: T[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    # This is NOT linear in the entries of U,V,W simultaneously. It's multilinear.
    # 
    # Better approach: The tensor condition is that the 3-tensor T equals the matmul tensor.
    # T[i,j,k] = sum_r U[i*m+j][r] V[j*p+k][r] W[k*n+i][r]
    # 
    # For a given sparsity pattern (which (row, col) pairs are nonzero in U,V,W), the values are 
    # determined by the equations. But this is a system of polynomial equations (cubic).
    #
    # Simpler approach: Just try removing a column (rank-1 term) and check if the remaining 
    # terms can still span. Since we're working with fixed integer structures, let's just 
    # try: remove column c, check if the resulting tensor is still valid. If not, try the next.
    # This is O(rank) checks, each O(n*m*p*rank) to compute tensor.
    
    # Actually, the simplest structural sparsification: try removing each column and verify.
    # If removing a column still gives a valid decomposition, keep it removed.
    # This finds rank reductions that come from linear dependencies in the rank-1 terms.
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        r = len(U[0]) if U else 0
        for c in range(r):
            if time.time() > deadline:
                break
            # Try removing column c
            U_new = [row[:c] + row[c+1:] for row in U]
            V_new = [row[:c] + row[c+1:] for row in V]
            W_new = [row[:c] + row[c+1:] for row in W]
            if _verify(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                changed = True
                break
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0

    # Start with the known rank-114 Strassen decomposition
    res = _build_strassen114(n, m, p)
    if not _verify(res["U"], res["V"], res["W"], n, m, p):
        # Fallback to schoolbook
        res = _build_schoolbook(n, m, p)
        if not _verify(res["U"], res["V"], res["W"], n, m, p):
            return res

    # Try structural sparsification: remove linearly dependent rank-1 terms
    if time.time() < deadline:
        res2 = _sparsify(res, n, m, p, deadline)
        if _verify(res2["U"], res2["V"], res2["W"], n, m, p):
            rank2 = len(res2["U"][0]) if res2["U"] else 0
            rank1 = len(res["U"][0]) if res["U"] else 0
            if rank2 < rank1:
                res = res2

    return res
