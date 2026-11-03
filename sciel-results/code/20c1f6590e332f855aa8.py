"""Matrix multiplication 2x5x5 rank-minimal exact tensor decomposition."""

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Block 1: M_{2,3,3} (i=0,1; j=0,1,2; k=0,1,2)
    # Rank 6: For each j in 0..2, for each i in 0..1
    col = 0
    for j in range(3):
        for i in range(2):
            # U[i*m + j] = 1
            # V[j*p + k] = 1 for k in 0..2
            # W[k*n + i] = 1 for k in 0..2
            add_term(i * m + j, 1, j * p + 0, 1, 0 * n + i, 1)
            add_term(i * m + j, 0, j * p + 1, 1, 1 * n + i, 1)
            add_term(i * m + j, 0, j * p + 2, 1, 2 * n + i, 1)
            col += 1

    # Block 2: M_{2,3,2} (i=0,1; j=0,1,2; k=3,4)
    # Rank 6: For each j in 0..2, for each i in 0..1
    for j in range(3):
        for i in range(2):
            # U[i*m + j] = 1
            # V[j*p + k] = 1 for k in 3..4
            # W[k*n + i] = 1 for k in 3..4
            add_term(i * m + j, 1, j * p + 3, 1, 3 * n + i, 1)
            add_term(i * m + j, 0, j * p + 4, 1, 4 * n + i, 1)
            col += 1

    # Block 3: M_{2,2,3} (i=0,1; j=3,4; k=0,1,2)
    # Rank 6: For each j in 3..4, for each i in 0..1
    for j in range(3, 5):
        for i in range(2):
            # U[i*m + j] = 1
            # V[j*p + k] = 1 for k in 0..2
            # W[k*n + i] = 1 for k in 0..2
            add_term(i * m + j, 1, j * p + 0, 1, 0 * n + i, 1)
            add_term(i * m + j, 0, j * p + 1, 1, 1 * n + i, 1)
            add_term(i * m + j, 0, j * p + 2, 1, 2 * n + i, 1)
            col += 1

    # Block 4: M_{2,2,2} (i=0,1; j=3,4; k=3,4)
    # Rank 7: Strassen decomposition
    # Let A be 2x2, B be 2x2, C = AB
    # Indices in local block: a_{i,j}, b_{j,k}, c_{i,k}
    # Global indices: i in 0..1, j in 3..4, k in 3..4
    # Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # c00 = M1+M4-M5+M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M6

    # Local to global mapping for Block 4:
    # a_{i,j} -> U[i*m + (j+3)]
    # b_{j,k} -> V[(j+3)*p + (k+3)]
    # c_{i,k} -> W[(k+3)*n + i]

    # Term 1: (a00+a11)(b00+b11)
    add_term(0 * m + 3, 1, 3 * p + 3, 1, 3 * n + 0, 1)
    add_term(1 * m + 4, 1, 3 * p + 3, 1, 3 * n + 0, 1)
    add_term(0 * m + 3, 1, 4 * p + 4, 1, 3 * n + 0, 1)
    add_term(1 * m + 4, 1, 4 * p + 4, 1, 3 * n + 0, 1)
    add_term(0 * m + 3, 1, 3 * p + 3, 1, 4 * n + 0, 1)
    add_term(1 * m + 4, 1, 3 * p + 3, 1, 4 * n + 0, 1)
    add_term(0 * m + 3, 1, 4 * p + 4, 1, 4 * n + 0, 1)
    add_term(1 * m + 4, 1, 4 * p + 4, 1, 4 * n + 0, 1)
    col += 1

    # Term 2: (a10+a11)b00
    add_term(1 * m + 3, 1, 3 * p + 3, 1, 4 * n + 0, 1)
    add_term(1 * m + 4, 1, 3 * p + 3, 1, 4 * n + 0, 1)
    add_term(1 * m + 3, 1, 3 * p + 3, 1, 4 * n + 1, 1)
    add_term(1 * m + 4, 1, 3 * p + 3, 1, 4 * n + 1, 1)
    col += 1

    # Term 3: a00(b10-b11)
    add_term(0 * m + 3, 1, 4 * p + 3, 1, 4 * n + 0, 1)
    add_term(0 * m + 3, 1, 4 * p + 4, -1, 4 * n + 0, 1)
    add_term(0 * m + 3, 1, 4 * p + 3, 1, 4 * n + 1, 1)
    add_term(0 * m + 3, 1, 4 * p + 4, -1, 4 * n + 1, 1)
    col += 1

    # Term 4: a11(b01-b00)
    add_term(1 * m + 4, 1, 3 * p + 4, 1, 3 * n + 0, 1)
    add_term(1 * m + 4, 1, 3 * p + 3, -1, 3 * n + 0, 1)
    add_term(1 * m + 4, 1, 3 * p + 4, 1, 4 * n + 0, 1)
    add_term(1 * m + 4, 1, 3 * p + 3, -1, 4 * n + 0, 1)
    col += 1

    # Term 5: (a00+a01)b11
    add_term(0 * m + 3, 1, 4 * p + 4, 1, 3 * n + 0, 1)
    add_term(0 * m + 4, 1, 4 * p + 4, 1, 3 * n + 0, 1)
    add_term(0 * m + 3, 1, 4 * p + 4, 1, 4 * n + 1, 1)
    add_term(0 * m + 4, 1, 4 * p + 4, 1, 4 * n + 1, 1)
    col += 1

    # Term 6: (a10-a00)(b00+b10)
    add_term(1 * m + 3, 1, 3 * p + 3, 1, 4 * n + 0, 1)
    add_term(0 * m + 3, -1, 3 * p + 3, 1, 4 * n + 0, 1)
    add_term(1 * m + 3, 1, 4 * p + 3, 1, 4 * n + 0, 1)
    add_term(0 * m + 3, -1, 4 * p + 3, 1, 4 * n + 0, 1)
    add_term(1 * m + 3, 1, 3 * p + 3, 1, 4 * n + 1, 1)
    add_term(0 * m + 3, -1, 3 * p + 3, 1, 4 * n + 1, 1)
    add_term(1 * m + 3, 1, 4 * p + 3, 1, 4 * n + 1, 1)
    add_term(0 * m + 3, -1, 4 * p + 3, 1, 4 * n + 1, 1)
    col += 1

    # Term 7: (a01-a11)(b01+b11)
    add_term(0 * m + 4, 1, 3 * p + 4, 1, 3 * n + 0, 1)
    add_term(1 * m + 4, -1, 3 * p + 4, 1, 3 * n + 0, 1)
    add_term(0 * m + 4, 1, 4 * p + 4, 1, 3 * n + 0, 1)
    add_term(1 * m + 4, -1, 4 * p + 4, 1, 3 * n + 0, 1)
    add_term(0 * m + 4, 1, 3 * p + 4, 1, 4 * n + 0, 1)
    add_term(1 * m + 4, -1, 3 * p + 4, 1, 4 * n + 0, 1)
    add_term(0 * m + 4, 1, 4 * p + 4, 1, 4 * n + 0, 1)
    add_term(1 * m + 4, -1, 4 * p + 4, 1, 4 * n + 0, 1)
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
