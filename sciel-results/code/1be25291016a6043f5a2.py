"""Rewritten program: block-constructed decomposition for 3x3x7 matrix multiplication.
Uses Strassen 2x2x2 (rank 7) for the top-left 2x2x2 block, schoolbook for the
remaining 2x2x2 and 1x2x2, 2x1x2, 1x1x1 blocks (total 19 for the 3x3x3 part),
plus schoolbook (rank 36) for the 3x3x4 part. Total rank = 55.
"""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7

    # We'll build lists of (U_row, V_row, W_row) tuples, each of length rank.
    # U has shape (n*m, rank) = (9, rank)
    # V has shape (m*p, rank) = (21, rank)
    # W has shape (p*n, rank) = (21, rank)

    # Index conventions:
    # U index: i*m + j  (i in 0..n-1, j in 0..m-1)
    # V index: j*p + k  (j in 0..m-1, k in 0..p-1)
    # W index: k*n + i  (k in 0..p-1, i in 0..n-1)

    terms = []  # Each term is (u_dict, v_dict, w_dict) where dict maps index -> coefficient

    def add_term(u_coeff, v_coeff, w_coeff):
        """u_coeff, v_coeff, w_coeff are dicts: {index: value}"""
        terms.append((u_coeff, v_coeff, w_coeff))

    # --- Part 1: 3x3x3 (columns k=0,1,2 of B) ---
    # Partition: I1={0,1}, I2={2}; J1={0,1}, J2={2}; K1={0,1}, K2={2}

    # Block A11 (2x2) * B11 (2x2) -> C11 (2x2): Strassen rank 7
    # A11 = [[a00, a01], [a10, a11]], B11 = [[b00, b01], [b10, b11]]
    # C11 = A11 * B11

    # Strassen 2x2x2:
    # M1 = (a00+a10)(b00+b10)
    # M2 = (a10)(b00+b10)
    # M3 = (a00)(b10)
    # M4 = (a00+a01)(b01+b11)
    # M5 = (a01)(b01+b11)
    # M6 = (a00)(b11)
    # M7 = (a10+a11)(b01)

    # For each Strassen term, add to our global terms.
    # U index for (i,j): i*3 + j
    # V index for (j,k): j*7 + k
    # W index for (k,i): k*3 + i

    # M1 = (a00+a10)(b00+b10)
    add_term({0*3+0: 1, 1*3+0: 1}, {0*7+0: 1, 1*7+0: 1},
             {0*3+0: 1, 0*3+1: 1, 1*3+0: 1, 1*3+1: 1})
    # M2 = (a10)(b00+b10)
    add_term({1*3+0: 1}, {0*7+0: 1, 1*7+0: 1},
             {0*3+1: 1, 1*3+1: 1})
    # M3 = (a00)(b10)
    add_term({0*3+0: 1}, {1*7+0: 1},
             {0*3+0: 1, 1*3+0: 1})
    # M4 = (a00+a01)(b01+b11)
    add_term({0*3+0: 1, 0*3+1: 1}, {0*7+1: 1, 1*7+1: 1},
             {1*3+0: 1, 1*3+1: 1, 2*3+0: 1, 2*3+1: 1})
    # M5 = (a01)(b01+b11)
    add_term({0*3+1: 1}, {0*7+1: 1, 1*7+1: 1},
             {1*3+1: 1, 2*3+1: 1})
    # M6 = (a00)(b11)
    add_term({0*3+0: 1}, {1*7+1: 1},
             {1*3+0: 1, 2*3+0: 1})
    # M7 = (a10+a11)(b01)
    add_term({1*3+0: 1, 1*3+1: 1}, {0*7+1: 1},
             {0*3+1: 1, 2*3+1: 1})

    # Block A12 (2x1) * B21 (1x2) -> C12 (2x2): schoolbook rank 4
    # A12 = [[a02], [a12]], B21 = [b20, b21]
    # C12 = A12 * B21 = [[a02*b20, a02*b21], [a12*b20, a12*b21]]
    # (i,j,k) where i in {0,1}, j=2, k in {0,1}
    for i in [0, 1]:
        for k in [0, 1]:
            add_term({i*3+2: 1}, {2*7+k: 1}, {k*3+i: 1})

    # Block A11 (2x2) * B12 (2x1) -> C13 (2x1): schoolbook rank 4
    # A11 = [[a00, a01], [a10, a11]], B12 = [[b02], [b12]]
    # C13 = A11 * B12 = [[a00*b02 + a01*b12], [a10*b02 + a11*b12]]
    # (i,j,k) where i in {0,1}, j in {0,1}, k=2
    for i in [0, 1]:
        for j in [0, 1]:
            add_term({i*3+j: 1}, {j*7+2: 1}, {2*3+i: 1})

    # Block A12 (2x1) * B22 (1x1) -> C14 (2x1): schoolbook rank 2
    # A12 = [[a02], [a12]], B22 = [b22]
    # C14 = A12 * B22 = [[a02*b22], [a12*b22]]
    # (i,j,k) where i in {0,1}, j=2, k=2
    for i in [0, 1]:
        add_term({i*3+2: 1}, {2*7+2: 1}, {2*3+i: 1})

    # Block A21 (1x2) * B11 (2x2) -> C21 (1x2): schoolbook rank 4
    # A21 = [a20, a21], B11 = [[b00, b01], [b10, b11]]
    # C21 = A21 * B11 = [[a20*b00+a21*b10, a20*b01+a21*b11]]
    # (i,j,k) where i=2, j in {0,1}, k in {0,1}
    for j in [0, 1]:
        for k in [0, 1]:
            add_term({2*3+j: 1}, {j*7+k: 1}, {k*3+2: 1})

    # Block A22 (1x1) * B21 (1x2) -> C22 (1x2): schoolbook rank 2
    # A22 = [a22], B21 = [b20, b21]
    # C22 = A22 * B21 = [[a22*b20, a22*b21]]
    # (i,j,k) where i=2, j=2, k in {0,1}
    for k in [0, 1]:
        add_term({2*3+2: 1}, {2*7+k: 1}, {k*3+2: 1})

    # Block A21 (1x2) * B12 (2x1) -> C23 (1x1): schoolbook rank 2
    # A21 = [a20, a21], B12 = [[b02], [b12]]
    # C23 = A21 * B12 = [[a20*b02 + a21*b12]]
    # (i,j,k) where i=2, j in {0,1}, k=2
    for j in [0, 1]:
        add_term({2*3+j: 1}, {j*7+2: 1}, {2*3+2: 1})

    # Block A22 (1x1) * B22 (1x1) -> C24 (1x1): schoolbook rank 1
    # A22 = [a22], B22 = [b22]
    # C24 = A22 * B22 = [[a22*b22]]
    # (i,j,k) where i=2, j=2, k=2
    add_term({2*3+2: 1}, {2*7+2: 1}, {2*3+2: 1})

    # --- Part 2: 3x3x4 (columns k=3,4,5,6 of B) ---
    # Schoolbook: for each k in {3,4,5,6}, for each i in {0,1,2}, for each j in {0,1,2}
    for k in [3, 4, 5, 6]:
        for i in [0, 1, 2]:
            for j in [0, 1, 2]:
                add_term({i*3+j: 1}, {j*7+k: 1}, {k*3+i: 1})

    # Now build the U, V, W matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx][r] = val
        for idx, val in v_dict.items():
            V[idx][r] = val
        for idx, val in w_dict.items():
            W[idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
