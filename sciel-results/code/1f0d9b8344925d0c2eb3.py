"""Matrix multiplication 2x4x5 rank-minimal decomposition."""
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5

    rank = 36
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    col = 0

    def ui(i, j):
        return i * m + j

    def vi(j, k):
        return j * p + k

    def wi(k, i):
        return k * n + i

    def add_term(u_entries, v_entries, w_entries):
        """Add one rank-1 term with given entries."""
        nonlocal col
        for (ui_row, val) in u_entries:
            U[ui_row][col] += val
        for (vi_row, val) in v_entries:
            V[vi_row][col] += val
        for (wi_row, val) in w_entries:
            W[wi_row][col] += val
        col += 1

    # Strassen 2x2x2: multiplies [[a,b],[c,d]] * [[e,f],[g,h]] -> [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # Maps (A row, col) -> (B row, col) -> (C row, col)
    def strassen(A_rows, B_rows, C_rows):
        """
        A_rows: [[(i0,j0),(i0,j1),(i1,j0),(i1,j1)]] - the 2x2 A block
        B_rows: [[(j0,k0),(j0,k1),(j1,k0),(j1,k1)]] - the 2x2 B block
        C_rows: [[(i0,k0),(i0,k1),(i1,k0),(i1,k1)]] - the 2x2 C block
        """
        a, b, c, d = A_rows       # a=(i0,j0), b=(i0,j1), c=(i1,j0), d=(i1,j1)
        e, f, g, h = B_rows       # e=(j0,k0), f=(j0,k1), g=(j1,k0), h=(j1,k1)
        ce, cf, cg, ch = C_rows   # ce=(i0,k0), cf=(i0,k1), cg=(i1,k0), ch=(i1,k1)

        # M1 = (a+d)(e+h) -> contributes to ce, ch (with ce getting +, ch getting +)
        # Actually M1 = (A00+A11)(B00+B11)
        # C00 += M1, C11 += M1
        add_term([(ui(*a), 1), (ui(*d), 1)],
                 [(vi(*e), 1), (vi(*h), 1)],
                 [(wi(*ce), 1), (wi(*ch), 1)])

        # M2 = (c+d)*e
        # C10 += M2, C11 -= M2
        add_term([(ui(*c), 1), (ui(*d), 1)],
                 [(vi(*e), 1)],
                 [(wi(*cg), 1), (wi(*ch), -1)])

        # M3 = a*(f-h)
        # C01 += M3, C11 += M3
        add_term([(ui(*a), 1)],
                 [(vi(*f), 1), (vi(*h), -1)],
                 [(wi(*cf), 1), (wi(*ch), 1)])

        # M4 = d*(g-e)
        # C00 += M4, C10 += M4
        add_term([(ui(*d), 1)],
                 [(vi(*g), 1), (vi(*e), -1)],
                 [(wi(*ce), 1), (wi(*cg), 1)])

        # M5 = (a+b)*h
        # C00 -= M5, C01 += M5
        add_term([(ui(*a), 1), (ui(*b), 1)],
                 [(vi(*h), 1)],
                 [(wi(*ce), -1), (wi(*cf), 1)])

        # M6 = (c-a)*(e+f)
        # C11 += M6
        add_term([(ui(*c), 1), (ui(*a), -1)],
                 [(vi(*e), 1), (vi(*f), 1)],
                 [(wi(*ch), 1)])

        # M7 = (b-d)*(g+h)
        # C00 += M7
        add_term([(ui(*b), 1), (ui(*d), -1)],
                 [(vi(*g), 1), (vi(*h), 1)],
                 [(wi(*ce), 1)])

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    strassen([(0,0), (0,1), (1,0), (1,1)],
             [(0,0), (0,1), (1,0), (1,1)],
             [(0,0), (0,1), (1,0), (1,1)])

    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    strassen([(0,2), (0,3), (1,2), (1,3)],
             [(2,0), (2,1), (3,0), (3,1)],
             [(0,0), (0,1), (1,0), (1,1)])

    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    strassen([(0,0), (0,1), (1,0), (1,1)],
             [(0,2), (0,3), (1,2), (1,3)],
             [(2,0), (2,1), (3,0), (3,1)])

    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    strassen([(0,2), (0,3), (1,2), (1,3)],
             [(2,2), (2,3), (3,2), (3,3)],
             [(2,0), (2,1), (3,0), (3,1)])

    # Block 5: A[0:2, 0:2] * B[0:2, 4] -> C[0:2, 4]
    # C[0,4] += A[0,0]*B[0,4] + A[0,1]*B[1,4]
    add_term([(ui(0,0), 1)], [(vi(0,4), 1)], [(wi(4,0), 1)])
    add_term([(ui(0,1), 1)], [(vi(1,4), 1)], [(wi(4,0), 1)])
    # C[1,4] += A[1,0]*B[0,4] + A[1,1]*B[1,4]
    add_term([(ui(1,0), 1)], [(vi(0,4), 1)], [(wi(4,1), 1)])
    add_term([(ui(1,1), 1)], [(vi(1,4), 1)], [(wi(4,1), 1)])

    # Block 6: A[0:2, 2:4] * B[2:4, 4] -> C[0:2, 4]
    # C[0,4] += A[0,2]*B[2,4] + A[0,3]*B[3,4]
    add_term([(ui(0,2), 1)], [(vi(2,4), 1)], [(wi(4,0), 1)])
    add_term([(ui(0,3), 1)], [(vi(3,4), 1)], [(wi(4,0), 1)])
    # C[1,4] += A[1,2]*B[2,4] + A[1,3]*B[3,4]
    add_term([(ui(1,2), 1)], [(vi(2,4), 1)], [(wi(4,1), 1)])
    add_term([(ui(1,3), 1)], [(vi(3,4), 1)], [(wi(4,1), 1)])

    # Trim to actual rank used
    if col < rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
