import sys
import time

def construct():
    """
    Rank-50 tensor decomposition for 2x4 by 4x7 matrix multiplication.
    Uses Strassen (rank 7) for 2x2x2 blocks and Schoolbook (rank 4) for 2x2x1 blocks.
    B(4x7) split into cols [0,1], [2,3], [4,5], [6].
    Each 4x2 block: 2 Strassen blocks (rank 14).
    The 4x1 block: 2 Schoolbook blocks (rank 8).
    Total: 3*14 + 8 = 50.
    """
    start_time = time.time()
    deadline = 120.0  # safety margin before 150s kill

    n, m, p = 2, 4, 7
    rank = 50

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = [0]  # use list for mutability in nested functions

    def add_strassen_block(A_indices, B_indices, C_indices):
        """Add a Strassen 2x2x2 decomposition (rank 7)."""
        # A: 2x2, B: 2x2, C: 2x2
        # Strassen 7 multiplications:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        def process_mk(A_coeffs, B_coeffs, C_coeffs):
            for local_a, coeff in A_coeffs.items():
                global_a = A_indices[local_a]
                U[global_a[0] * m + global_a[1]][col[0]] += coeff
            for local_b, coeff in B_coeffs.items():
                global_b = B_indices[local_b]
                V[global_b[0] * p + global_b[1]][col[0]] += coeff
            for local_c, coeff in C_coeffs.items():
                global_c = C_indices[local_c]
                W[global_c[1] * n + global_c[0]][col[0]] += coeff
            col[0] += 1

        # M1 = (a00+a11)(b00+b11)
        process_mk({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1})
        # M2 = (a10+a11)b00
        process_mk({2:1, 3:1}, {0:1}, {2:1, 3:-1})
        # M3 = a00(b01-b11)
        process_mk({0:1}, {1:1, 3:-1}, {1:1, 3:1})
        # M4 = a11(b10-b00)
        process_mk({3:1}, {2:1, 0:-1}, {0:1, 2:1})
        # M5 = (a00+a01)b11
        process_mk({0:1, 1:1}, {3:1}, {0:-1, 1:1})
        # M6 = (a10-a00)(b00+b01)
        process_mk({2:1, 0:-1}, {0:1, 1:1}, {3:1})
        # M7 = (a01-a11)(b10+b11)
        process_mk({1:1, 3:-1}, {2:1, 3:1}, {0:1})

    def add_schoolbook_221(A_indices, B_indices, C_indices):
        """Add a schoolbook 2x2x1 decomposition (rank 4)."""
        # A: 2x2, B: 2x1, C: 2x1
        def process_mult(A_idx, B_idx, C_idx, coeff=1):
            global_a = A_indices[A_idx]
            global_b = B_indices[B_idx]
            global_c = C_indices[C_idx]
            U[global_a[0] * m + global_a[1]][col[0]] += coeff
            V[global_b[0] * p + global_b[1]][col[0]] += coeff
            W[global_c[1] * n + global_c[0]][col[0]] += coeff
            col[0] += 1

        process_mult(0, 0, 0, 1)
        process_mult(1, 1, 0, 1)
        process_mult(2, 0, 1, 1)
        process_mult(3, 1, 1, 1)

    # A_L: rows 0,1 of A (indices 0,1 in A's row-major)
    A_L = [(0,0), (0,1), (1,0), (1,1)]
    # A_R: rows 2,3 of A
    A_R = [(0,2), (0,3), (1,2), (1,3)]

    # Block 1: B cols 0,1 (4x2)
    B_1_L = [(0,0), (0,1), (1,0), (1,1)]
    B_1_R = [(2,0), (2,1), (3,0), (3,1)]
    C_1 = [(0,0), (0,1), (1,0), (1,1)]

    add_strassen_block(A_L, B_1_L, C_1)
    add_strassen_block(A_R, B_1_R, C_1)

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Block 2: B cols 2,3 (4x2)
    B_2_L = [(0,2), (0,3), (1,2), (1,3)]
    B_2_R = [(2,2), (2,3), (3,2), (3,3)]
    C_2 = [(0,2), (0,3), (1,2), (1,3)]

    add_strassen_block(A_L, B_2_L, C_2)
    add_strassen_block(A_R, B_2_R, C_2)

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Block 3: B cols 4,5 (4x2)
    B_3_L_L = [(0,4), (0,5), (1,4), (1,5)]
    B_3_L_R = [(2,4), (2,5), (3,4), (3,5)]
    C_3_L = [(0,4), (0,5), (1,4), (1,5)]

    add_strassen_block(A_L, B_3_L_L, C_3_L)
    add_strassen_block(A_R, B_3_L_R, C_3_L)

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Block 4: B col 6 (4x1)
    B_4_L = [(0,6), (1,6)]
    B_4_R = [(2,6), (3,6)]
    C_4 = [(0,6), (1,6)]

    add_schoolbook_221(A_L, B_4_L, C_4)
    add_schoolbook_221(A_R, B_4_R, C_4)

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
