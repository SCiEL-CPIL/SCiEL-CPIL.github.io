"""Matrix multiplication 2x5x5 rank-minimal exact tensor decomposition."""

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49  # 7 (Strassen for 2x2) + 3 (A_2 B_21) + 6 (A_1 B_12) + 33 (A_2 B_22)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0

    # Strassen's 7 products for the 2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1})
    # U indices: i*m + j, V indices: j*p + k, W indices: k*n + i
    strassen = [
        ([0, 1, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1]),  # P1: (A00+A11)(B00+B11)
        ([0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 1, 0]),  # P2: (A10+A11)B00
        ([1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]),  # P3: A00(B01-B11)
        ([0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1]),  # P4: A11(B10-B00)
        ([1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]),  # P5: (A00+A01)B11
        ([1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]),  # P6: (A10-A00)(B00+B01)
        ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 0]),  # P7: (A01-A11)(B10+B11)
    ]

    for u_vec, v_vec, w_vec in strassen:
        # u_vec corresponds to A indices: [A00, A01, A10, A11]
        # Global U indices: [0*5+0, 0*5+1, 1*5+0, 1*5+1] = [0, 1, 5, 6]
        u_indices = [0, 1, 5, 6]
        for idx, val in zip(u_indices, u_vec):
            U[idx][column] += val

        # v_vec corresponds to B indices: [B00, B01, B10, B11]
        # Global V indices: [0*5+0, 0*5+1, 1*5+0, 1*5+1] = [0, 1, 5, 6]
        v_indices = [0, 1, 5, 6]
        for idx, val in zip(v_indices, v_vec):
            V[idx][column] += val

        # w_vec corresponds to C indices: [C00, C01, C10, C11]
        # Global W indices: [0*2+0, 1*2+0, 0*2+1, 1*2+1] = [0, 2, 1, 3]
        w_indices = [0, 2, 1, 3]
        for idx, val in zip(w_indices, w_vec):
            W[idx][column] += val

        column += 1

    # Trivial terms for A_2 B_21: i in {0,1}, j in {2,3,4}, k in {0,1}
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    # Trivial terms for A_1 B_12: i in {0,1}, j in {0,1}, k in {2,3,4}
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    # Trivial terms for A_2 B_22: i in {0,1}, j in {2,3,4}, k in {2,3,4}
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
