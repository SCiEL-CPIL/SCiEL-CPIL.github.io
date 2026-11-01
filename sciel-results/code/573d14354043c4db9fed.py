"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    r_count = 7
    U = [[0] * r_count for _ in range(n * m)]
    V = [[0] * r_count for _ in range(m * p)]
    W = [[0] * r_count for _ in range(p * n)]

    def set_val(M, row, col, val):
        M[row][col] += val

    # r=0: (a+b)(c+d)
    set_val(U, 0, 0, 1); set_val(U, 3, 0, 1)
    set_val(V, 0, 0, 1); set_val(V, 3, 0, 1)
    set_val(W, 0, 0, 1); set_val(W, 3, 0, 1)

    # r=1: (b+c)a
    set_val(U, 2, 1, 1); set_val(U, 3, 1, 1)
    set_val(V, 0, 1, 1)
    set_val(W, 1, 1, 1); set_val(W, 3, 1, -1)

    # r=2: (a-d)(c+e)
    set_val(U, 0, 2, 1)
    set_val(V, 1, 2, 1); set_val(V, 3, 2, -1)
    set_val(W, 2, 2, 1); set_val(W, 3, 2, 1)

    # r=3: (b-d)(d+e)
    set_val(U, 3, 3, 1)
    set_val(V, 2, 3, 1); set_val(V, 0, 3, -1)
    set_val(W, 0, 3, 1); set_val(W, 1, 3, 1)

    # r=4: a(b-e)
    set_val(U, 0, 4, 1); set_val(U, 1, 4, 1)
    set_val(V, 3, 4, 1)
    set_val(W, 0, 4, -1); set_val(W, 2, 4, 1)

    # r=5: c(a-b)
    set_val(U, 2, 5, 1); set_val(U, 0, 5, -1)
    set_val(V, 0, 5, 1); set_val(V, 1, 5, 1)
    set_val(W, 3, 5, 1)

    # r=6: e(b-c)
    set_val(U, 1, 6, 1); set_val(U, 3, 6, -1)
    set_val(V, 2, 6, 1); set_val(V, 3, 6, 1)
    set_val(W, 0, 6, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
