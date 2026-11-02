"""Exact rank-minimal tensor decomposition for 2x4 * 4x6 matrix multiplication.
Structural change: exploit the direct-sum structure of the multiplication tensor
along the column index of B.  The 2x4 * 4x6 tensor decomposes as three copies
of the 2x4 * 4x2 tensor (one per pair of columns of B).  We search for a
rank-12 decomposition of the 2x4 * 4x2 sub-tensor using exact rational
arithmetic with iterative bilinear linearisation.  If found, total rank = 36.
Otherwise we fall back to the known rank-42 Strassen construction, then to
schoolbook rank-48.
"""
import time
import numpy as np
from fractions import Fraction

N, M, P = 2, 4, 6
NM, MP, PN = N * M, M * P, P * N  # 8, 24, 12
START = time.time()
DEADLINE = 135.0  # safety margin before 150s kill


def construct():
    # Strategy:
    #   1. Try a structured exact-rank-36 construction (three rank-12 sub-decomps).
    #   2. Fall back to the rank-42 Strassen construction (known correct).
    #   3. Fall back to schoolbook rank-48.
    try:
        if _time_ok():
            U, V, W = _try_rank36()
            if U is not None and _verify_fast(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    try:
        if _time_ok():
            U, V, W = _strassen42()
            if _verify_fast(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass

    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _time_ok():
    return (time.time() - START) < DEADLINE


# ---------------------------------------------------------------------------
# Known rank-42 Strassen construction (2x2 blocks, 6 independent Strassen
# decompositions for the 2x2 * 2x6 sub-problems).
# ---------------------------------------------------------------------------

def _strassen42():
    rank = 42
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))

    # Strassen coefficients for 2x2 matrices.
    S = [
        [1, 0, 1, 0],   # A00 + A11
        [0, 0, 1, 1],   # A10 + A11
        [1, 0, 0, 0],   # A00
        [0, 0, 0, 1],   # A11
        [1, 1, 0, 0],   # A00 + A01
        [-1, 0, 1, 0],  # A10 - A00
        [0, -1, 0, 1]   # A01 - A11
    ]
    T = [
        [1, 0, 0, 1],   # B00 + B11
        [1, 0, 0, 0],   # B00
        [0, 1, 0, -1],  # B01 - B11
        [0, 0, 1, -1],  # B10 - B00
        [0, 0, 0, 1],   # B11
        [1, 1, 0, 0],   # B00 + B01
        [0, 0, 1, 1]    # B10 + B11
    ]
    R = [
        [1, 0, 0, -1, 1, 0, 1],   # C00
        [0, 0, 1, 1, 0, 0, 0],    # C01
        [0, 1, 0, 0, 1, 0, 0],    # C10
        [1, 0, -1, 0, 0, 1, 1]    # C11
    ]

    # 6 blocks: (A_col_off, B_row_off, B_col_off, C_col_off)
    blocks = [
        (0, 0, 0, 0),
        (0, 0, 2, 2),
        (0, 0, 4, 4),
        (2, 2, 0, 0),
        (2, 2, 2, 2),
        (2, 2, 4, 4)
    ]

    col = 0
    for a_co, b_ro, b_co, c_co in blocks:
        for k in range(7):
            for i in range(2):
                for jl in range(2):
                    jg = a_co + jl
                    U[i * M + jg, col] += S[k][i * 2 + jl]
            for jl in range(2):
                jg = b_ro + jl
                for kl in range(2):
                    kg = b_co + kl
                    V[jg * P + kg, col] += T[k][jl * 2 + kl]
            for il in range(2):
                ig = il
                for kl in range(2):
                    kg = c_co + kl
                    W[kg * N + ig, col] += R[il * 2 + kl][k]
            col += 1
    return U, V, W


# ---------------------------------------------------------------------------
# Schoolbook rank-48 fallback.
# ---------------------------------------------------------------------------

def _schoolbook():
    rank = NM * M * P // M  # n*m*p = 48
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, c] = 1
                V[j * P + k, c] = 1
                W[k * N + i, c] = 1
                c += 1
    return U, V, W


# ---------------------------------------------------------------------------
# Exact rank-36 construction via three rank-12 decompositions of 2x4*4x2.
# ---------------------------------------------------------------------------

def _try_rank36():
    """Try to construct a rank-36 decomposition using three rank-12
    decompositions of the 2x4*4x2 sub-problems."""
    if not _time_ok():
        return None, None, None

    U12, V12, W12 = _find_rank12_2x4x4x2()
    if U12 is None:
        return None, None, None

    # Assemble the full rank-36 decomposition from three copies.
    rank = 36
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))

    # Three column-pair offsets in B: 0, 2, 4
    col_offsets = [0, 2, 4]
    for t, co in enumerate(col_offsets):
        base = t * 12
        for u in range(NM):
            U[u, base:base+12] = U12[u, :]
        for v in range(MP):
            # V index: row = j (0..3), col = k (0..5)
            # In the sub-problem, B is 4x2, so k_sub = 0,1
            # Global k = co + k_sub
            # V12 index: row = j, col = k_sub -> V12[j*2 + k_sub]
            # Global V index: j*P + k = j*6 + co + k_sub
            for j in range(M):
                for k_sub in range(2):
                    v_sub = j * 2 + k_sub
                    v_glob = j * P + co + k_sub
                    V[v_glob, base:base+12] = V12[v_sub, :]
        for w in range(PN):
            # W index: row = k (0..5), col = i (0..1)
            # In the sub-problem, k_sub = 0,1
            # Global k = co + k_sub
            # W12 index: row = k_sub, col = i -> W12[k_sub*2 + i]
            # Global W index: k*N + i = (co + k_sub)*2 + i
            for i in range(N):
                for k_sub in range(2):
                    w_sub = k_sub * 2 + i
                    w_glob = (co + k_sub) * N + i
                    W[w_glob, base:base+12] = W12[w_sub, :]

    return U, V, W


_RANK12_CACHE = None

def _find_rank12_2x4x4x2():
    """Find a rank-12 decomposition of the 2x4 * 4x2 multiplication tensor.
    Uses iterative bilinear linearisation with exact rational arithmetic."""
    global _RANK12_CACHE
    if _RANK12_CACHE is not None:
        return _RANK12_CACHE
    if not _time_ok():
        return None

    n, m, p = 2, 4, 2
    nm, mp, pn = n * m, m * p, p * n  # 8, 8, 4
    r = 12

    # Build the tensor: T[u,v,w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i else 0
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[(u, v, w)] = Fraction(1)

    # Initialize U, V, W with Strassen structure for two 2x2 blocks (rank 14),
    # then compress to rank 12 via iterative refinement.
    U = [[Fraction(0)] * r for _ in range(nm)]
    V = [[Fraction(0)] * r for _ in range(mp)]
    W = [[Fraction(0)] * r for _ in range(pn)]

    S = [
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, -1, 0, 1]
    ]
    Tc = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    Rc = [
        [1, 0, 0, -1, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [1, 0, -1, 0, 0, 1, 1]
    ]

    # Initialize with Strassen for two 2x2 blocks (columns 0-1 and 2-3 of A,
    # rows 0-1 and 2-3 of B).
    for blk, (a_off, b_off) in enumerate([(0, 0), (2, 2)]):
        for k in range(7):
            col = blk * 7 + k
            if col >= r:
                break
            for i in range(2):
                for jl in range(2):
                    jg = a_off + jl
                    U[i * m + jg][col] = Fraction(S[k][i * 2 + jl])
            for jl in range(2):
                jg = b_off + jl
                for kl in range(2):
                    kg = kl
                    V[jg * p + kg][col] = Fraction(Tc[k][jl * 2 + kl])
            for il in range(2):
                ig = il
                for kl in range(2):
                    kg = kl
                    W[kg * n + ig][col] = Fraction(Rc[il * 2 + kl][k])

    # Iterative bilinear linearisation: alternate solving for U, V, W.
    # At each step, fix two matrices and solve a linear system for the third.
    # We use least-squares (exact rational) to minimize the residual.
    
    best_err = None
    best_U, best_V, best_W = None, None, None

    for iteration in range(200):
        if not _time_ok():
            break

        # Solve for U: fix V, W.
        # For each (v, w): sum_s U[u,s] * V[v,s] * W[w,s] = T[u,v,w] for all u.
        # This gives nm equations in nm*r unknowns. But the coefficient vector
        # c_s = V[v,s] * W[w,s] is shared across u. So for each (v,w), we have:
        #   U[:, s] dot c = T[:, v, w]  (vector equation of length nm)
        # This is underdetermined. Use least-squares.
        
        # Actually, let's think differently. The full tensor equation is:
        #   sum_s U[:,s] outer V[:,s] outer W[:,s] = T
        # For fixed V, W, this is linear in U. We can vectorize:
        #   For each s, the contribution to T from term s is:
        #     T_s[u,v,w] = U[u,s] * V[v,s] * W[w,s]
        #   So T = sum_s U[:,s] * (V[:,s] * W[:,s])^T  (in some sense)
        #
        # Let's use a different approach: treat U as unknown, and for each
        # (v,w), the equation is:
        #   sum_s U[u,s] * (V[v,s] * W[w,s]) = T[u,v,w]  for each u
        # This is a system: for each (v,w), we have a linear map from
        # U[:,s] (nm*r variables) to T[:,v,w] (nm values).
        # The coefficient matrix is nm x (nm*r), with entries:
        #   A[u, u'*r + s] = delta_{u,u'} * V[v,s] * W[w,s]
        # This is block-diagonal with blocks of size 1 x r.
        # So for each u, we have: sum_s U[u,s] * V[v,s] * W[w,s] = T[u,v,w]
        # This is r equations (one per (v,w)) in r unknowns (U[u,s]).
        # We can solve this for each u independently!
        
        # For each u, solve: sum_s U[u,s] * V[v,s] * W[w,s] = T[u,v,w] for all (v,w)
        # This is a system of (mp*pn) equations in r unknowns.
        # mp*pn = 8*4 = 32 > r = 12, so overdetermined. Use least-squares.
        
        new_U = [[Fraction(0)] * r for _ in range(nm)]
        for u in range(nm):
            # Build the system: for each (v,w), equation: sum_s U[u,s] * V[v,s] * W[w,s] = T[u,v,w]
            # Coefficient vector for (v,w): c_s = V[v,s] * W[w,s]
            # We want to find x = U[u,:] such that A x = b, where
            # A[row, s] = V[v_row, s] * W[w_row, s], b[row] = T[u, v_row, w_row]
            # Use normal equations: A^T A x = A^T b
            # A^T A is r x r, A^T b is r x 1.
            ATA = [[Fraction(0)] * r for _ in range(r)]
            ATb = [Fraction(0)] * r
            for v in range(mp):
                for w in range(pn):
                    tw = T.get((u, v, w), Fraction(0))
                    for s1 in range(r):
                        coeff = V[v][s1] * W[w][s1]
                        ATb[s1] += coeff * tw
                        for s2 in range(r):
                            ATA[s1][s2] += coeff * V[v][s2] * W[w][s2]
            # Solve ATA x = ATb
            try:
                sol = _solve_rational(ATA, ATb)
                if sol is not None:
                    new_U[u] = sol
            except Exception:
                pass
        
        U = new_U

        # Solve for V: fix U, W.
        new_V = [[Fraction(0)] * r for _ in range(mp)]
        for v in range(mp):
            ATA = [[Fraction(0)] * r for _ in range(r)]
            ATb = [Fraction(0)] * r
            for u in range(nm):
                for w in range(pn):
                    tw = T.get((u, v, w), Fraction(0))
                    for s1 in range(r):
                        coeff = U[u][s1] * W[w][s1]
                        ATb[s1] += coeff * tw
                        for s2 in range(r):
                            ATA[s1][s2] += coeff * U[u][s2] * W[w][s2]
            try:
                sol = _solve_rational(ATA, ATb)
                if sol is not None:
                    new_V[v] = sol
            except Exception:
                pass
        V = new_V

        # Solve for W: fix U, V.
        new_W = [[Fraction(0)] * r for _ in range(pn)]
        for w in range(pn):
            ATA = [[Fraction(0)] * r for _ in range(r)]
            ATb = [Fraction(0)] * r
            for u in range(nm):
                for v in range(mp):
                    tw = T.get((u, v, w), Fraction(0))
                    for s1 in range(r):
                        coeff = U[u][s1] * V[v][s1]
                        ATb[s1] += coeff * tw
                        for s2 in range(r):
                            ATA[s1][s2] += coeff * U[u][s2] * V[v][s2]
            try:
                sol = _solve_rational(ATA, ATb)
                if sol is not None:
                    new_W[w] = sol
            except Exception:
                pass
        W = new_W

        # Check error
        err = _tensor_error(U, V, W, T, nm, mp, pn, r)
        if best_err is None or err < best_err:
            best_err = err
            best_U = [row[:] for row in U]
            best_V = [row[:] for row in V]
            best_W = [row[:] for row in W]
        if err == 0:
            break

    if best_err is not None and best_err == 0:
        _RANK12_CACHE = (
            np.array([[float(x) for x in row] for row in best_U]),
            np.array([[float(x) for x in row] for row in best_V]),
            np.array([[float(x) for x in row] for row in best_W])
        )
        return _RANK12_CACHE

    # If we didn't get exact zero, check if it's close enough (shouldn't happen
    # with exact arithmetic, but just in case).
    _RANK12_CACHE = None
    return None


def _tensor_error(U, V, W, T, nm, mp, pn, r):
    """Compute the maximum absolute error of the tensor decomposition."""
    max_err = Fraction(0)
    for u in range(nm):
        for v in range(mp):
            for w in range(pn):
                val = sum(U[u][s] * V[v][s] * W[w][s] for s in range(r))
                target = T.get((u, v, w), Fraction(0))
                err = abs(val - target)
                if err > max_err:
                    max_err = err
    return max_err


def _solve_rational(A, b):
    """Solve Ax = b for x using Gaussian elimination over Q.
    A is r x r, b is r. Returns x or None if singular."""
    n = len(b)
    # Augmented matrix
    M = [[A[i][j] for j in range(n)] + [b[i]] for i in range(n)]
    
    # Forward elimination
    for col in range(n):
        # Find pivot
        pivot = None
        for row in range(col, n):
            if M[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            return None  # Singular
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]
        # Eliminate below
        inv_pivot = 1 / M[col][col]
        M[col] = [x * inv_pivot for x in M[col]]
        for row in range(col + 1, n):
            factor = M[row][col]
            if factor != 0:
                M[row] = [M[row][j] - factor * M[col][j] for j in range(n + 1)]
    
    # Back substitution
    x = [Fraction(0)] * n
    for row in range(n - 1, -1, -1):
        s = M[row][n]
        for j in range(row + 1, n):
            s -= M[row][j] * x[j]
        x[row] = s
    return x


def _verify_fast(U, V, W):
    """Verify the decomposition by checking all tensor entries."""
    nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)

    for u in range(nm):
        i = u // M
        j = u % M
        for v in range(mp):
            jv = v // P
            k = v % P
            if jv != j:
                # Should be zero for all w
                prod = U[u, :] * V[v, :]
                for w in range(pn):
                    if abs(np.dot(prod, W[w, :])) > 1e-7:
                        return False
            else:
                w_t = k * N + i
                prod = U[u, :] * V[v, :]
                if abs(np.dot(prod, W[w_t, :]) - 1.0) > 1e-7:
                    return False
                for w in range(pn):
                    if w == w_t:
                        continue
                    if abs(np.dot(prod, W[w, :])) > 1e-7:
                        return False
    return True


if __name__ == "__main__":
    result = construct()
    print(f"Best rank found: {len(result['U'][0])}")
