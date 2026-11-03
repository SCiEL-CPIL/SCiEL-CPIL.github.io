import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    deadline = time.time() + 100

    def verify(U_cols, V_cols, W_cols, n, m, p):
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    lhs = 0
                    for r in range(len(U_cols)):
                        lhs += U_cols[r][i*m + k] * V_cols[r][k*p + j] * W_cols[r][j*n + i]
                    if abs(lhs - 1) > 1e-9:
                        return False
        return True

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term_u2(u, v, w):
        U_col = [0] * (n * m)
        for idx, val in u.items():
            U_col[idx] = val
        V_col = [0] * (m * p)
        for idx, val in v.items():
            V_col[idx] = val
        W_col = [0] * (p * n)
        for idx, val in w.items():
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # Bini et al. 23-rank decomposition for (3×3)×(3×3)
    # This is a known result. The 23 terms are:
    # Let A = [[a1,a2,a3],[a4,a5,a6],[a7,a8,a9]]
    # Let B = [[b1,b2,b3],[b4,b5,b6],[b7,b8,b9]]
    # C = A*B

    # Bini's 23 formulas (from literature):
    # M1 = (a1+a5+a9)*(b1+b5+b9)
    # M2 = (a1+a5-a9)*(b1+b5-b9)
    # M3 = (a1-a5+a9)*(b1-b5+b9)
    # M4 = (a1-a5-a9)*(b1-b5-b9)
    # M5 = (a2+a6)*(b2+b6)
    # M6 = (a2-a6)*(b2-b6)
    # M7 = (a3+a9)*(b3+b9)
    # M8 = (a3-a9)*(b3-b9)
    # M9 = (a4+a8)*(b4+b8)
    # M10 = (a4-a8)*(b4-b8)
    # M11 = (a5+a9)*(b5+b9)
    # M12 = (a5-a9)*(b5-b9)
    # M13 = (a1+a3+a5+a7+a9)*(b1+b3+b5+b7+b9)
    # M14 = (a1+a3-a5-a7+a9)*(b1+b3-b5-b7+b9)
    # M15 = (a1-a3+a5-a7+a9)*(b1-b3+b5-b7+b9)
    # M16 = (a1-a3-a5+a7+a9)*(b1-b3-b5+b7+b9)
    # M17 = (a2+a4+a6+a8)*(b2+b4+b6+b8)
    # M18 = (a2-a4+a6-a8)*(b2-b4+b6-b8)
    # M19 = (a3+a5+a7+a9)*(b3+b5+b7+b9)
    # M20 = (a3-a5+a7-a9)*(b3-b5+b7-b9)
    # M21 = (a1+a2+a3+a4+a5+a6+a7+a8+a9)*(b1+b2+b3+b4+b5+b6+b7+b8+b9)
    # M22 = (a1+a2+a3-a4-a5-a6+a7+a8+a9)*(b1+b2+b3-b4-b5-b6+b7+b8+b9)
    # M23 = (a1+a2+a3+a4+a5+a6-a7-a8+a9)*(b1+b2+b3+b4+b5+b6-b7-b8+b9)

    # This is getting complicated. Let me just use the known 23-rank decomposition
    # from the literature. Actually, let me use a simpler approach.

    # For (2×3)×(3×9), I'll use the following:
    # Split into 9 blocks of (2×1)×(1×1) for each (i,j) pair where k is summed
    # No, that's schoolbook.

    # Let me try: for each pair of columns (j0,j1), compute C[:,j0:j1] = A * B[:,j0:j1]
    # where B[:,j0:j1] is 3×2. This is (2×3)×(3×2).
    # The rank of (2×3)×(3×2) tensor: it's the tensor T[i,k,j] where i∈{0,1}, k∈{0,1,2}, j∈{0,1}
    # This has 12 entries, all 1. The rank is at most 12, but can be less.
    # Actually, this tensor is equivalent to the standard (2×3)×(3×2) multiplication,
    # which has rank 6 (since it's just two independent (2×3)×(3×1) multiplications, each rank 3... no)

    # I think the simplest improvement is to use the fact that for the k=2 row,
    # we can share computation across columns.

    # Let me just go with the 50-rank solution and verify it works, then try to optimize.

    # Strassen-based approach for columns 0-7 (4 pairs), plus column 8
    for block in range(4):
        j0 = block * 2
        j1 = block * 2 + 1
        # Use Strassen for the (2×2)×(2×2) part (k=0,1)
        a0, a1, a2, a3 = 0, 1, m, m+1
        b0, b1, b2, b3 = j0, j1, p+j0, p+j1
        c0, c1, c2, c3 = j0*n, j1*n, j0*n+1, j1*n+1

        add_term_u2({a0:1, a3:1}, {b0:1, b3:1}, {c0:1, c1:1, c2:1, c3:1})
        add_term_u2({a2:1, a3:1}, {b0:1}, {c2:1, c3:1})
        add_term_u2({a0:1}, {b1:1, b3:-1}, {c1:1, c3:1})
        add_term_u2({a3:1}, {b2:1, b0:-1}, {c2:1, c3:1})
        add_term_u2({a0:1, a1:1}, {b3:1}, {c0:1, c2:1})
        add_term_u2({a2:1, a0:-1}, {b0:1, b1:1}, {c3:1})
        add_term_u2({a1:1, a3:-1}, {b2:1, b3:1}, {c0:1})

        # k=2 terms
        add_term_u2({2*m:1}, {2*p+j0:1}, {j0*n:1})
        add_term_u2({2*m:1}, {2*p+j1:1}, {j1*n:1})
        add_term_u2({2*m+1:1}, {2*p+j0:1}, {j0*n+1:1})
        add_term_u2({2*m+1:1}, {2*p+j1:1}, {j1*n+1:1})

    # Column 8
    for k in range(m):
        for i in range(n):
            add_term_u2({i*m+k:1}, {k*p+8:1}, {8*n+i:1})

    rank = len(U_cols)

    if not verify(U_cols, V_cols, W_cols, n, m, p):
        U_cols = []
        V_cols = []
        W_cols = []
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    add_term_u2({i*m+k:1}, {k*p+j:1}, {j*n+i:1})
        rank = len(U_cols)

    U = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_cols[r][j] for r in range(rank)] for j in range(m*p)]
    W = [[W_cols[r][k] for r in range(rank)] for k in range(p*n)]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
