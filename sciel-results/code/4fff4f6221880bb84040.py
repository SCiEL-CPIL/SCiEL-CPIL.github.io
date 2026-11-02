"""Rank-minimal exact tensor decomposition for 2x3 x 3x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()
_DEADLINE = 120.0
_SAFETY = 8.0  # seconds margin before evaluator kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # We need: for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # W uses trace convention: index (k*n + i).

    # Theoretical lower bound on rank.
    lb = max(n * m, m * p, p * n)
    # Known upper bound: schoolbook n*m*p.
    ub = n * m * p

    def deadline_reached():
        return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY)

    def try_rank(r, rng, max_iter=200000):
        """Try to find integer/half-integer U,V,W of rank r. Returns dict or None."""
        # We fix U and V, solve for W per (i,k) block.
        # For fixed (i,k): 3 equations (over j) in r unknowns w_c:
        #   sum_c U[i*m+j,c]*V[j*p+k,c]*w_c = 1  for j=0..m-1
        for it in range(max_iter):
            if deadline_reached():
                return None
            # Random small integer init for U and V.
            scale = 2
            U = rng.randint(-scale, scale, size=(n * m, r))
            V = rng.randint(-scale, scale, size=(m * p, r))
            # Ensure some non-trivial structure: make first column nonzero-ish.
            U[0, 0] = 1
            V[0, 0] = 1

            W = np.zeros((p * n, r), dtype=object)
            ok = True
            for i in range(n):
                for k in range(p):
                    # Build 3 x r matrix M where M[j,c] = U[i*m+j,c]*V[j*p+k,c]
                    M = np.zeros((m, r), dtype=object)
                    for j in range(m):
                        for c in range(r):
                            M[j, c] = U[i * m + j, c] * V[j * p + k, c]
                    # Solve M w = 1 (3 eqs, r unknowns). Need exact rational solution.
                    # Use a particular solution: pick r-3 free vars = 0, solve 3x3 if possible.
                    # Choose 3 pivot columns with nonzero determinant.
                    sol = _solve_small(M, m, r)
                    if sol is None:
                        ok = False
                        break
                    W[k * n + i, :] = sol
                if not ok:
                    break
            if not ok:
                continue
            # Verify all equations exactly.
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    def _solve_small(M, m, r):
        """Solve M w = 1 over rationals, m eqs, r unknowns, m <= 3.
        Returns integer/half-integer solution or None."""
        # Find m pivot columns with nonzero determinant.
        import itertools
        cols = list(range(r))
        for combo in itertools.combinations(cols, m):
            sub = M[:, list(combo)]
            det = _det(sub, m)
            if det != 0:
                # Cramer's rule for the m pivots, free vars = 0.
                sol = [0] * r
                for a, c in enumerate(combo):
                    sub_c = sub.copy()
                    sub_c[:, a] = 1  # RHS
                    sol[c] = _det(sub_c, m) / det
                # Convert to half-integers: multiply by 2, check integrality.
                doubled = [2 * x for x in sol]
                if all(_is_int(d) for d in doubled):
                    return [d / 2 for d in doubled]
                return None
        return None

    def _det(A, s):
        """Exact integer determinant of s x s (s<=3)."""
        if s == 1:
            return A[0, 0]
        if s == 2:
            return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if s == 3:
            return (A[0,0]*(A[1,1]*A[2,2]-A[1,2]*A[2,1])
                    - A[0,1]*(A[1,0]*A[2,2]-A[1,2]*A[2,0])
                    + A[0,2]*(A[1,0]*A[2,1]-A[1,1]*A[2,0]))
        raise ValueError

    def _is_int(x):
        return abs(x - round(x)) < 1e-9

    def _verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(U.shape[1]):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    rng = random.Random(12345)
    # Try ranks from lower bound up.
    for r in range(lb, ub + 1):
        if deadline_reached():
            break
        res = try_rank(r, rng)
        if res is not None:
            return res
        # If we've exhausted search for this rank, move up.

    # Fallback: schoolbook (always valid), rank n*m*p.
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
# EVOLVE-BLOCK-END
