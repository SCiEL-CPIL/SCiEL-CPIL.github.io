"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Strategy:
    # Split m=3 into m=2 (j in {0,1}) and m=1 (j=2).
    # m=2, p=8: split p=8 into four p=2 blocks, each 2x2x2 Strassen (rank 7). Total 28.
    # m=1, p=8: T[i,2,k] = A[i,2]*B[2,k] is a rank-1 tensor (outer product). Total 1.
    # Grand total rank: 29.
    
    rank = 29
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 29
    V = [[0] * rank for _ in range(m * p)]  # 24 x 29
    W = [[0] * rank for _ in range(p * n)]  # 16 x 29
    
    col = 0
    
    def add_strassen_2x2x2(i0, j0, k0):
        """Add Strassen rank-7 decomposition for 2x2x2 at offsets (i0,j0,k0)."""
        nonlocal col
        # m1: (a+d)(e+h)
        U[(i0+0)*m + (j0+0)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+0)*p + (k0+0)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+0)*n + (i0+0)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        # m2: (c+d)e
        U[(i0+1)*m + (j0+0)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+0)*p + (k0+0)][col] += 1
        W[(k0+0)*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += -1
        col += 1
        # m3: a(f-h)
        U[(i0+0)*m + (j0+0)][col] += 1
        V[(j0+0)*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += -1
        W[(k0+1)*n + (i0+0)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        # m4: d(g-e)
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+0)][col] += 1
        V[(j0+0)*p + (k0+0)][col] += -1
        W[(k0+0)*n + (i0+0)][col] += 1
        W[(k0+0)*n + (i0+1)][col] += 1
        col += 1
        # m5: (a+b)h
        U[(i0+0)*m + (j0+0)][col] += 1
        U[(i0+0)*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+0)*n + (i0+0)][col] += -1
        W[(k0+1)*n + (i0+0)][col] += 1
        col += 1
        # m6: (c-a)(e+f)
        U[(i0+1)*m + (j0+0)][col] += 1
        U[(i0+0)*m + (j0+0)][col] += -1
        V[(j0+0)*p + (k0+0)][col] += 1
        V[(j0+0)*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        # m7: (b-d)(g+h)
        U[(i0+0)*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += -1
        V[(j0+1)*p + (k0+0)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+0)*n + (i0+0)][col] += 1
        col += 1

    # m=2 part: four 2x2x2 Strassen blocks for j in {0,1}, k in {0..7}
    for k_block in range(4):
        add_strassen_2x2x2(0, 0, k_block * 2)
    
    # m=1 part: rank-1 for j=2, all k
    # U[i*3+2, col] = 1, V[2*8+k, col] = 1, W[k*2+i, col] = 1
    for i in range(n):
        U[i*m + 2][col] = 1
    for k in range(p):
        V[2*p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k*n + i][col] = 1
    col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
