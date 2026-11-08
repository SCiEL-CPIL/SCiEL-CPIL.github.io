"""Rewritten program for matrix multiplication 2,4,5 with rank-36 decomposition."""

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    import time
    t_start = time.time()
    DEADLINE = 110.0  # safety margin before 120s limit

    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)

    rank = 36

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0

    def strassen_block(y_indices, z_indices, w_offsets):
        """
        Perform 7 scalar multiplications for a 2x2 x 2x2 Strassen block.
        
        y_indices: [(y00_i, y00_j), (y01_i, y01_j), (y10_i, y10_j), (y11_i, y11_j)]
                   These are the (row, col) indices in A for the 4 quadrants.
                   y00 = A[0,0] block (top-left), y01 = A[0,1] block (top-right),
                   y10 = A[1,0] block (bottom-left), y11 = A[1,1] block (bottom-right)
        
        z_indices: [(z00_i, z00_j), (z01_i, z01_j), (z10_i, z10_j), (z11_i, z11_j)]
                   These are the (row, col) indices in B for the 4 quadrants.
                   z00 = B[0,0] block, z01 = B[0,1] block,
                   z10 = B[1,0] block, z11 = B[1,1] block
        
        w_offsets: [(w00_k, w00_i), (w10_k, w10_i), (w01_k, w01_i), (w11_k, w11_i)]
                   These are the (col_in_B, row_in_C) indices for the 4 output quadrants.
                   w00 = C[0,0], w10 = C[1,0], w01 = C[0,1], w11 = C[1,1]
        """
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        w00, w10, w01, w11 = w_offsets

        if time.time() > DEADLINE:
            return

        # M1: (A00 + A11)(B00 + B11) -> contributes to C00 and C11
        U[u_idx(y00[0], y00[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w00[0], w00[1])][col] += 1
        W[w_idx(w11[0], w11[1])][col] += 1
        col += 1

        # M2: (A10 + A11)B00 -> contributes to C10 and C11
        U[u_idx(y10[0], y10[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] += 1
        W[w_idx(w10[0], w10[1])][col] += 1
        W[w_idx(w11[0], w11[1])][col] -= 1
        col += 1

        # M3: A00(B01 - B11) -> contributes to C01 and C11
        U[u_idx(y00[0], y00[1])][col] += 1
        V[v_idx(z01[0], z01[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] -= 1
        W[w_idx(w01[0], w01[1])][col] += 1
        W[w_idx(w11[0], w11[1])][col] += 1
        col += 1

        # M4: A11(B10 - B00) -> contributes to C00 and C10
        U[u_idx(y11[0], y11[1])][col] += 1
        V[v_idx(z10[0], z10[1])][col] += 1
        V[v_idx(z00[0], z00[1])][col] -= 1
        W[w_idx(w00[0], w00[1])][col] += 1
        W[w_idx(w10[0], w10[1])][col] += 1
        col += 1

        # M5: (A00 + A01)B11 -> contributes to C00 and C01
        U[u_idx(y00[0], y00[1])][col] += 1
        U[u_idx(y01[0], y01[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w00[0], w00[1])][col] -= 1
        W[w_idx(w01[0], w01[1])][col] += 1
        col += 1

        # M6: (A10 - A00)(B00 + B01) -> contributes to C11
        U[u_idx(y10[0], y10[1])][col] += 1
        U[u_idx(y00[0], y00[1])][col] -= 1
        V[v_idx(z00[0], z00[1])][col] += 1
        V[v_idx(z01[0], z01[1])][col] += 1
        W[w_idx(w11[0], w11[1])][col] += 1
        col += 1

        # M7: (A01 - A11)(B10 + B11) -> contributes to C11
        U[u_idx(y01[0], y01[1])][col] += 1
        U[u_idx(y11[0], y11[1])][col] -= 1
        V[v_idx(z10[0], z10[1])][col] += 1
        V[v_idx(z11[0], z11[1])][col] += 1
        W[w_idx(w11[0], w11[1])][col] += 1
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A quadrants: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    # B quadrants: B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1)
    # C quadrants: C00 k=0,i=0; C10 k=0,i=1; C01 k=1,i=0; C11 k=1,i=1
    y1 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    z1 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    w1 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    strassen_block(y1, z1, w1)

    if time.time() > DEADLINE:
        return {"U": U, "V": V, "W": W}

    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A quadrants: A00=(0,2), A01=(0,3), A10=(1,2), A11=(1,3)
    # B quadrants: B00=(2,0), B01=(2,1), B10=(3,0), B11=(3,1)
    # C quadrants: same as Block 1
    y2 = [(0, 2), (0, 3), (1, 2), (1, 3)]
    z2 = [(2, 0), (2, 1), (3, 0), (3, 1)]
    w2 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    strassen_block(y2, z2, w2)

    if time.time() > DEADLINE:
        return {"U": U, "V": V, "W": W}

    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A quadrants: same as Block 1
    # B quadrants: B00=(0,2), B01=(0,3), B10=(1,2), B11=(1,3)
    # C quadrants: C00 k=2,i=0; C10 k=2,i=1; C01 k=3,i=0; C11 k=3,i=1
    y3 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    z3 = [(0, 2), (0, 3), (1, 2), (1, 3)]
    w3 = [(2, 0), (2, 1), (3, 0), (3, 1)]
    strassen_block(y3, z3, w3)

    if time.time() > DEADLINE:
        return {"U": U, "V": V, "W": W}

    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # A quadrants: same as Block 2
    # B quadrants: B00=(2,2), B01=(2,3), B10=(3,2), B11=(3,3)
    # C quadrants: same as Block 3
    y4 = [(0, 2), (0, 3), (1, 2), (1, 3)]
    z4 = [(2, 2), (2, 3), (3, 2), (3, 3)]
    w4 = [(2, 0), (2, 1), (3, 0), (3, 1)]
    strassen_block(y4, z4, w4)

    if time.time() > DEADLINE:
        return {"U": U, "V": V, "W": W}

    # Now handle the last column: A[0:2, 0:4] * B[0:4, 4:5] -> C[0:2, 4:5]
    # This is a 2x4 x 4x1 multiplication.
    # We can split this into two 2x2 x 2x1 multiplications.
    # Sub-block 5: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    # Sub-block 6: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]

    # For a 2x2 x 2x1 multiplication, we need 4 scalar multiplications.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]

    # Sub-block 5: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    # A: (0,0), (0,1), (1,0), (1,1)
    # B: (0,4), (1,4)
    # C: (4,0), (4,1)

    # M5_1: A[0,0]*B[0,4] -> C[4,0]
    U[u_idx(0, 0)][col] += 1
    V[v_idx(0, 4)][col] += 1
    W[w_idx(4, 0)][col] += 1
    col += 1

    # M5_2: A[0,1]*B[1,4] -> C[4,0]
    U[u_idx(0, 1)][col] += 1
    V[v_idx(1, 4)][col] += 1
    W[w_idx(4, 0)][col] += 1
    col += 1

    # M5_3: A[1,0]*B[0,4] -> C[4,1]
    U[u_idx(1, 0)][col] += 1
    V[v_idx(0, 4)][col] += 1
    W[w_idx(4, 1)][col] += 1
    col += 1

    # M5_4: A[1,1]*B[1,4] -> C[4,1]
    U[u_idx(1, 1)][col] += 1
    V[v_idx(1, 4)][col] += 1
    W[w_idx(4, 1)][col] += 1
    col += 1

    if time.time() > DEADLINE:
        return {"U": U, "V": V, "W": W}

    # Sub-block 6: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]
    # A: (0,2), (0,3), (1,2), (1,3)
    # B: (2,4), (3,4)
    # C: (4,0), (4,1)

    # M6_1: A[0,2]*B[2,4] -> C[4,0]
    U[u_idx(0, 2)][col] += 1
    V[v_idx(2, 4)][col] += 1
    W[w_idx(4, 0)][col] += 1
    col += 1

    # M6_2: A[0,3]*B[3,4] -> C[4,0]
    U[u_idx(0, 3)][col] += 1
    V[v_idx(3, 4)][col] += 1
    W[w_idx(4, 0)][col] += 1
    col += 1

    # M6_3: A[1,2]*B[2,4] -> C[4,1]
    U[u_idx(1, 2)][col] += 1
    V[v_idx(2, 4)][col] += 1
    W[w_idx(4, 1)][col] += 1
    col += 1

    # M6_4: A[1,3]*B[3,4] -> C[4,1]
    U[u_idx(1, 3)][col] += 1
    V[v_idx(3, 4)][col] += 1
    W[w_idx(4, 1)][col] += 1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
