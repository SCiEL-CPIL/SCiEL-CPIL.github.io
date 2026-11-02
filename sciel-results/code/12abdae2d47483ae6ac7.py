"""Matrix multiplication 3x3x4 rank-minimal decomposition."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 4
    rank = 34
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    def add_ua(i, j, c, val):
        U[i * m + j][c] += val
    def add_va(j, k, c, val):
        V[j * p + k][c] += val
    def add_wa(k, i, c, val):
        W[k * n + i][c] += val

    def strassen_block(c0, i_off, j_off, k_off):
        # col 0: (0,0),(0,0),(0,0) all +1
        add_ua(i_off+0, j_off+0, c0+0, 1)
        add_va(j_off+0, k_off+0, c0+0, 1)
        add_wa(k_off+0, i_off+0, c0+0, 1)
        # col 1: (1,0),(0,1),(1,1) all +1
        add_ua(i_off+1, j_off+0, c0+1, 1)
        add_va(j_off+0, k_off+1, c0+1, 1)
        add_wa(k_off+1, i_off+1, c0+1, 1)
        # col 2: (1,1),(1,1),(1,1) W=-1
        add_ua(i_off+1, j_off+1, c0+2, 1)
        add_va(j_off+1, k_off+1, c0+2, 1)
        add_wa(k_off+1, i_off+1, c0+2, -1)
        # col 3: (0,1)+(1,0), (0,0), (0,0) W=-1
        add_ua(i_off+0, j_off+1, c0+3, 1)
        add_ua(i_off+1, j_off+0, c0+3, 1)
        add_va(j_off+0, k_off+0, c0+3, 1)
        add_wa(k_off+0, i_off+0, c0+3, -1)
        # col 4: (0,1)-(1,0), (0,1), (1,0) all +1
        add_ua(i_off+0, j_off+1, c0+4, -1)
        add_ua(i_off+1, j_off+0, c0+4, -1)
        add_va(j_off+0, k_off+1, c0+4, 1)
        add_wa(k_off+1, i_off+0, c0+4, 1)
        # col 5: (0,0)-(0,1), (1,0)+(1,1), (0,1) all +1
        add_ua(i_off+0, j_off+0, c0+5, 1)
        add_ua(i_off+0, j_off+1, c0+5, -1)
        add_va(j_off+1, k_off+0, c0+5, 1)
        add_va(j_off+1, k_off+1, c0+5, 1)
        add_wa(k_off+1, i_off+0, c0+5, 1)
        # col 6: (0,1)-(1,1), (0,0)+(0,1), (1,0) W=-1
        add_ua(i_off+0, j_off+1, c0+6, -1)
        add_ua(i_off+1, j_off+1, c0+6, -1)
        add_va(j_off+0, k_off+0, c0+6, 1)
        add_va(j_off+0, k_off+1, c0+6, 1)
        add_wa(k_off+1, i_off+0, c0+6, -1)

    # Block A: i∈{0,1}, j∈{0,1}, k∈{0,1} — cols 0-6
    strassen_block(0, 0, 0, 0)
    # Block B: i∈{0,1}, j∈{0,1}, k∈{2,3} — cols 7-13
    strassen_block(7, 0, 0, 2)
    # Block C: i∈{0,1}, j=2, k∈{0,1,2,3} — cols 14-21
    c = 14
    for ii in range(2):
        for kk in range(4):
            add_ua(ii, 2, c, 1)
            add_va(2, kk, c, 1)
            add_wa(kk, ii, c, 1)
            c += 1
    # Block D: i=2, j∈{0,1,2}, k∈{0,1,2,3} — cols 22-33
    c = 22
    for jj in range(3):
        for kk in range(4):
            add_ua(2, jj, c, 1)
            add_va(jj, kk, c, 1)
            add_wa(kk, 2, c, 1)
            c += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
