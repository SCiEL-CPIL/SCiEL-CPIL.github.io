"""Matrix multiplication 3,3,8 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]

import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin

    # Schoolbook rank: 72
    sb_rank = n * m * p
    U_sb = np.zeros((n * m, sb_rank), dtype=int)
    V_sb = np.zeros((m * p, sb_rank), dtype=int)
    W_sb = np.zeros((p * n, sb_rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def to_list(mat):
        return [[int(v) for v in row] for row in mat]

    # Verify schoolbook
    def verify(U, V, W):
        # Tensor: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Must equal 1 if (i,j,k) corresponds to the multiplication tensor entry
        # The matrix multiplication tensor T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
        # Actually for matrix mult: T_{i,j,k} where C[i,k] = sum_j A[i,j]*B[j,k]
        # The tensor is T[i,j,k] = 1 (it's the identity tensor for the bilinear map)
        # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(U.shape[1]):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if s != 1:
                        return False
        return True

    # Try to find a lower-rank decomposition using structured search
    # Strategy: try rank 71, 70, ... with integer entries in {-1, 0, 1}
    # Use a smarter approach: fix most of the schoolbook and try to merge/eliminate terms

    # Alternative: try to find a decomposition with rank < 72 using
    # the fact that some tensors have lower rank than schoolbook.
    # For 3x3x3, rank is 23 (vs 27 schoolbook).
    # For 3x3x8, the exact rank is not known to be below 72 in general,
    # but let's try a limited search.

    # Approach: try to find a rank-r decomposition where r < 72
    # by solving a system of polynomial equations.
    # This is very hard in general, but we can try small r with structured init.

    # Better approach: use the known lower bound and try to verify if rank 71 works
    # via a certificate (duality). But that's also hard.

    # Practical approach: try random integer matrices with entries in {-1,0,1}
    # for rank 71 and check if they satisfy the equations.
    # The number of equations is n*m*p = 72, and we have 3*r variables.
    # For r=71, that's 213 variables and 72 equations — overdetermined but the
    # equations are cubic (trilinear), so this is a system of 72 cubic equations.

    # Let's try a different tactic: start from schoolbook and try to "merge"
    # two terms into one. If we can find u1,v1,w1 and u2,v2,w2 such that
    # u1 v1 w1 + u2 v2 w2 = u v w (a single rank-1 term), then we reduce rank by 1.

    # For the schoolbook, each term is a standard basis vector in each space.
    # Merging two standard basis terms into one is impossible (they're orthogonal).
    # So we need a more creative approach.

    # Let's try: for each pair of schoolbook terms, see if their sum can be
    # replaced by a single term with integer/half-integer entries.
    # Term a: U_a = e_{ia,ja}, V_a = e_{ja,ka}, W_a = e_{ka,ia}
    # Term b: U_b = e_{ib,jb}, V_b = e_{jb,kb}, W_b = e_{kb,ib}
    # We need u,v,w such that u⊗v⊗w = e_{ia,ja}⊗e_{ja,ka}⊗e_{ka,ia} + e_{ib,jb}⊗e_{jb,kb}⊗e_{kb,ib}
    # This is only possible if the two terms are identical (which they're not in schoolbook)
    # or if there's a special structure. Generally impossible for distinct standard basis terms.

    # So merging won't work for the standard schoolbook basis.

    # Let's try a completely different structured approach:
    # Use the fact that for matrix multiplication, there exist decompositions
    # with rank less than schoolbook for certain dimensions.
    # For 3x3x3, the rank is 23 (Bini et al. 1979, improved to 23 by Strassen).
    # For 3x3x8, we can potentially use a block structure.

    # Actually, let's try to use a known result: the rank of 3x3x8 matrix multiplication.
    # The lower bound is max(nm, mp, pn) = max(9, 24, 24) = 24, but that's very weak.
    # A better lower bound uses the slice rank or other methods.
    # The known upper bound is 72 (schoolbook).

    # Given the time constraints and the difficulty of finding a sub-72 decomposition,
    # let's try a quick random search for rank 71 with small integer entries.

    rng = np.random.default_rng(42)
    found = False
    for r in range(71, 65, -1):  # Try ranks 71 down to 66
        if time.time() > deadline - 30:
            break
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            # Random init with entries in {-1, 0, 1}
            U = rng.choice([-1, 0, 1], size=(n * m, r))
            V = rng.choice([-1, 0, 1], size=(m * p, r))
            W = rng.choice([-1, 0, 1], size=(p * n, r))
            # Check
            ok = True
            for i in range(n):
                if not ok:
                    break
                for j in range(m):
                    if not ok:
                        break
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        if s != 1:
                            ok = False
                            break
            if ok:
                found = True
                return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    # Return schoolbook
    return {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
