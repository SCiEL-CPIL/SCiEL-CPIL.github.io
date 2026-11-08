import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    rank = 34
    U = [[0] * rank for _ in range(n * m)]  # 8 x 34
    V = [[0] * rank for _ in range(m * p)]  # 24 x 34
    W = [[0] * rank for _ in range(p * n)]  # 12 x 34

    # Index helpers
    def a_idx(i, j):
        return i * m + j

    def b_idx(i, j):
        return i * p + j

    def c_idx(i, j):
        # W[k*n+i, r] where k is col of C, i is row of C
        return j * n + i

    col = 0

    # --- Part 1: A1 * B11 (2x2 times 2x2), rank 7 Strassen ---
    # A1 = A[:, 0:2], B11 = B[0:2, 0:2]
    # C contribution: C[0:2, 0:2]
    for group in [1, 2]:
        # group 1: A1, B11 (B rows 0,1; B cols 0,1)
        # group 2: A2, B21 (B rows 2,3; B cols 0,1)
        a_off = (group - 1) * 2  # row offset for A
        b_row_off = (group - 1) * 2  # row offset for B
        # B col offset is 0 for both groups (B11 and B21 are both cols 0,1)
        b_col_off = 0
        c_col_off = 0  # C cols 0,1

        # P1: (A11+A22)(B11+B22) -> C11+1, C22+1
        U[a_idx(a_off + 0, 0)][col] += 1
        U[a_idx(a_off + 1, 1)][col] += 1
        V[b_idx(b_row_off + 0, b_col_off + 0)][col] += 1
        V[b_idx(b_row_off + 1, b_col_off + 1)][col] += 1
        W[c_idx(0, c_col_off + 0)][col] += 1
        W[c_idx(1, c_col_off + 1)][col] += 1
        col += 1

        # P2: (A21+A22)B11 -> C21+1, C22-1
        U[a_idx(a_off + 1, 0)][col] += 1
        U[a_idx(a_off + 1, 1)][col] += 1
        V[b_idx(b_row_off + 0, b_col_off + 0)][col] += 1
        W[c_idx(1, c_col_off + 0)][col] += 1
        W[c_idx(1, c_col_off + 1)][col] -= 1
        col += 1

        # P3: A11(B12-B22) -> C12+1, C22+1
        U[a_idx(a_off + 0, 0)][col] += 1
        V[b_idx(b_row_off + 0, b_col_off + 1)][col] += 1
        V[b_idx(b_row_off + 1, b_col_off + 1)][col] -= 1
        W[c_idx(0, c_col_off + 1)][col] += 1
        W[c_idx(1, c_col_off + 1)][col] += 1
        col += 1

        # P4: A22(B21-B11) -> C11+1, C21+1
        U[a_idx(a_off + 1, 1)][col] += 1
        V[b_idx(b_row_off + 1, b_col_off + 0)][col] += 1
        V[b_idx(b_row_off + 0, b_col_off + 0)][col] -= 1
        W[c_idx(0, c_col_off + 0)][col] += 1
        W[c_idx(1, c_col_off + 0)][col] += 1
        col += 1

        # P5: (A11+A12)B22 -> C11-1, C12+1
        U[a_idx(a_off + 0, 0)][col] += 1
        U[a_idx(a_off + 0, 1)][col] += 1
        V[b_idx(b_row_off + 1, b_col_off + 1)][col] += 1
        W[c_idx(0, c_col_off + 0)][col] -= 1
        W[c_idx(0, c_col_off + 1)][col] += 1
        col += 1

        # P6: (A21-A11)(B11+B12) -> C22+1
        U[a_idx(a_off + 1, 0)][col] += 1
        U[a_idx(a_off + 0, 0)][col] -= 1
        V[b_idx(b_row_off + 0, b_col_off + 0)][col] += 1
        V[b_idx(b_row_off + 0, b_col_off + 1)][col] += 1
        W[c_idx(1, c_col_off + 1)][col] += 1
        col += 1

        # P7: (A12-A22)(B21+B22) -> C11+1
        U[a_idx(a_off + 0, 1)][col] += 1
        U[a_idx(a_off + 1, 1)][col] -= 1
        V[b_idx(b_row_off + 1, b_col_off + 0)][col] += 1
        V[b_idx(b_row_off + 1, b_col_off + 1)][col] += 1
        W[c_idx(0, c_col_off + 0)][col] += 1
        col += 1

    # --- Part 2: A1 * (B12 + B13) (2x2 times 2x4), rank 10 ---
    # A1 = A[:, 0:2], B_right = B[0:2, 2:6]
    # C contribution: C[0:2, 2:6]
    # Rank 10 decomposition for 2x2 x 2x4
    # Let B = [B_left B_right] where B_left is 2x2, B_right is 2x2
    # Actually B_right is 2x4, so let's split into two 2x2 blocks: B12 and B13
    # C = A1 * B12 + A1 * B13
    # We use a rank 10 algorithm for multiplying 2x2 by 2x4.
    # One known rank-10 decomposition:
    # Let A = [[a, b], [c, d]], B = [[e, f, g, h], [i, j, k, l]]
    # C = A*B = [[ae+bi, af+bj, ag+bk, ah+bl], [ce+di, cf+dj, cg+dk, ch+dl]]
    # 
    # Rank 10 decomposition (from literature, e.g., Bini et al.):
    # p1 = (a+c)(e+i)
    # p2 = (a+d)(e+j)
    # p3 = (a+b)(e+k)
    # p4 = (a+c)(f+i)
    # p5 = (a+d)(f+j)
    # p6 = (a+b)(f+k)
    # p7 = (c-d)(g+h)
    # p8 = (c-d)(k+l)
    # p9 = (b-d)(g+h)
    # p10 = (b-d)(k+l)
    # 
    # Hmm, let me use a simpler approach. Since 2x2x4 has rank 10,
    # I'll use two overlapping 2x2x2 Strassen decompositions with shared terms.
    # Actually, let me just use two separate 2x2x2 Strassen (rank 7 each) = rank 14.
    # That gives total rank 7+7+7+14 = 35. Still better than 42.
    # Or I can use rank 10 if I find the right coefficients.
    # 
    # Let me use a known rank 10 decomposition for 2x2 x 2x4.
    # Reference: The rank of 2x2x4 tensor is 10.
    # 
    # Let me use a simpler valid decomposition: just two Strassen (rank 14).
    # Total: 7 + 7 + 7 + 14 = 35. Score = 48/35 = 1.371. Better than 1.1429.
    # 
    # Actually wait, let me recount. Current is 42. 
    # If I use: A1*B11 (7) + A1*B12 (7) + A1*B13 (7) + A2*B21 (7) + A2*B22 (7) + A2*B23 (7) = 42. That's the current.
    # 
    # Better: A1*B11 (7) + A1*(B12+B13) (10) + A2*B21 (7) + A2*(B22+B23) (10) = 34.
    # 
    # Let me implement rank 10 for 2x2x4 properly.
    # 
    # Known rank 10 decomposition for M_{2,2,4}:
    # Using the decomposition from Landsberg or similar.
    # 
    # Let me use a different valid approach: 
    # A1 * B12 (2x2 x 2x2, rank 7) + A1 * B13 (2x2 x 2x2, rank 7) = 14
    # But we can share some structure.
    # 
    # For now, let me just do two separate Strassen for the right 2x4 block.
    # That gives rank 14 for each A_i * (B_i2 + B_i3).
    # Total: 7 + 14 + 7 + 14 = 42. Same as before.
    # 
    # I need the rank 10 decomposition. Let me look it up mentally.
    # 
    # For M_{2,2,4}, one rank 10 decomposition:
    # Let A = [[a, b], [c, d]], B = [[e, f, g, h], [i, j, k, l]]
    # 
    # The 10 products can be:
    # t1 = (a+c)(e+i)
    # t2 = (a+c)(f+j)  
    # t3 = (a+d)(e+i)
    # t4 = (a+d)(f+j)
    # t5 = (b-d)(g+h)
    # t6 = (b-d)(k+l)
    # t7 = (b-d)(e+i)
    # t8 = (b-d)(f+j)
    # t9 = (a-c)(g+h)
    # t10 = (a-c)(k+l)
    # 
    # This is getting complex. Let me just use two Strassen blocks (rank 14) for simplicity.
    # Total rank = 7 + 7 + 7 + 7 + 7 + 7 = 42. No improvement.
    # 
    # OK let me just implement the two-Strassen approach for the 2x4 part, giving rank 14 each.
    # Total: 7 + 14 + 7 + 14 = 42. No improvement.
    # 
    # I'll stick with the current approach but verify it's correct. The current score is 1.1429 = 48/42.
    # Let me just return the same solution to be safe.

    # Actually, let me just implement the current working solution correctly.
    # Reset and use the original approach.
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def get_a_idx(group, local_i, local_j):
        global_col = local_j + 2 * (group - 1)
        global_row = local_i
        return global_row * m + global_col

    def get_b_idx(group, local_i, local_j, block_col):
        global_row = local_i + 2 * (group - 1)
        global_col = local_j + 2 * block_col
        return global_row * p + global_col

    def get_c_idx(local_i, local_j, block_col):
        global_row = local_i
        global_col = local_j + 2 * block_col
        return global_col * n + global_row

    col = 0
    for group in [1, 2]:
        for block_col in [0, 1, 2]:
            # P1
            U[get_a_idx(group, 0, 0)][col] += 1
            U[get_a_idx(group, 1, 1)][col] += 1
            V[get_b_idx(group, 0, 0, block_col)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] += 1
            W[get_c_idx(0, 0, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] += 1
            col += 1

            # P2
            U[get_a_idx(group, 1, 0)][col] += 1
            U[get_a_idx(group, 1, 1)][col] += 1
            V[get_b_idx(group, 0, 0, block_col)][col] += 1
            W[get_c_idx(1, 0, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] -= 1
            col += 1

            # P3
            U[get_a_idx(group, 0, 0)][col] += 1
            V[get_b_idx(group, 0, 1, block_col)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] -= 1
            W[get_c_idx(0, 1, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] += 1
            col += 1

            # P4
            U[get_a_idx(group, 1, 1)][col] += 1
            V[get_b_idx(group, 1, 0, block_col)][col] += 1
            V[get_b_idx(group, 0, 0, block_col)][col] -= 1
            W[get_c_idx(0, 0, block_col)][col] += 1
            W[get_c_idx(1, 0, block_col)][col] += 1
            col += 1

            # P5
            U[get_a_idx(group, 0, 0)][col] += 1
            U[get_a_idx(group, 0, 1)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] += 1
            W[get_c_idx(0, 0, block_col)][col] -= 1
            W[get_c_idx(0, 1, block_col)][col] += 1
            col += 1

            # P6
            U[get_a_idx(group, 1, 0)][col] += 1
            U[get_a_idx(group, 0, 0)][col] -= 1
            V[get_b_idx(group, 0, 0, block_col)][col] += 1
            V[get_b_idx(group, 0, 1, block_col)][col] += 1
            W[get_c_idx(1, 1, block_col)][col] += 1
            col += 1

            # P7
            U[get_a_idx(group, 0, 1)][col] += 1
            U[get_a_idx(group, 1, 1)][col] -= 1
            V[get_b_idx(group, 1, 0, block_col)][col] += 1
            V[get_b_idx(group, 1, 1, block_col)][col] += 1
            W[get_c_idx(0, 0, block_col)][col] += 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
