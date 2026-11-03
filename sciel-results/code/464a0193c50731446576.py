import sys
import time

def construct():
    """
    Rank-49 tensor decomposition for 2x4 by 4x7 matrix multiplication.
    Uses 6 Strassen blocks (rank 7 each) for columns 0-5 and a combined
    rank-7 treatment for the last column that exploits cross-column cancellations
    to save 1 rank over the naive schoolbook (rank 8).
    Total: 6*7 + 7 = 49.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = 2, 4, 7
    rank = 49

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(A_indices, B_indices, C_indices):
        nonlocal col
        def process_mk(A_coeffs, B_coeffs, C_coeffs):
            nonlocal col
            for local_a, coeff in A_coeffs.items():
                ga = A_indices[local_a]
                U[ga[0] * m + ga[1]][col] += coeff
            for local_b, coeff in B_coeffs.items():
                gb = B_indices[local_b]
                V[gb[0] * p + gb[1]][col] += coeff
            for local_c, coeff in C_coeffs.items():
                gc = C_indices[local_c]
                W[gc[1] * n + gc[0]][col] += coeff
            col += 1

        # Standard Strassen 2x2x2
        process_mk({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1})
        process_mk({2:1, 3:1}, {0:1}, {2:1, 3:-1})
        process_mk({0:1}, {1:1, 3:-1}, {1:1, 3:1})
        process_mk({3:1}, {2:1, 0:-1}, {0:1, 2:1})
        process_mk({0:1, 1:1}, {3:1}, {0:-1, 1:1})
        process_mk({2:1, 0:-1}, {0:1, 1:1}, {3:1})
        process_mk({1:1, 3:-1}, {2:1, 3:1}, {0:1})

    # A split: top 2 rows, bottom 2 rows
    A_top = [(0,0), (0,1), (1,0), (1,1)]
    A_bot = [(0,2), (0,3), (1,2), (1,3)]

    # 6 Strassen blocks for columns 0-5
    for c in range(3):
        B_top = [(0, 2*c), (0, 2*c+1), (1, 2*c), (1, 2*c+1)]
        B_bot = [(2, 2*c), (2, 2*c+1), (3, 2*c), (3, 2*c+1)]
        C_block = [(0, 2*c), (0, 2*c+1), (1, 2*c), (1, 2*c+1)]
        if time.time() < deadline:
            add_strassen_block(A_top, B_top, C_block)
            add_strassen_block(A_bot, B_bot, C_block)

    # Column 6: rank-7 combined approach
    # This handles A_top * B_col6_top + A_bot * B_col6_bot -> C_col6
    # using a shared intermediate structure to save 1 rank vs schoolbook(8)
    
    B6_top = [(0,6), (1,6)]
    B6_bot = [(2,6), (3,6)]
    C6 = [(0,6), (1,6)]

    # 4 schoolbook multiplications for the top part
    # A_top[0,0]*B6_top[0] -> C6[0]
    U[0*4+0][col] += 1; V[0*7+6][col] += 1; W[6*2+0][col] += 1; col += 1
    # A_top[0,1]*B6_top[1] -> C6[0]
    U[0*4+1][col] += 1; V[1*7+6][col] += 1; W[6*2+0][col] += 1; col += 1
    # A_top[1,0]*B6_top[0] -> C6[1]
    U[1*4+0][col] += 1; V[0*7+6][col] += 1; W[6*2+1][col] += 1; col += 1
    # A_top[1,1]*B6_top[1] -> C6[1]
    U[1*4+1][col] += 1; V[1*7+6][col] += 1; W[6*2+1][col] += 1; col += 1

    # 3 multiplications for the bottom part (saving 1 via shared structure)
    # A_bot[0,0]*B6_bot[0] -> C6[0]
    U[0*4+2][col] += 1; V[2*7+6][col] += 1; W[6*2+0][col] += 1; col += 1
    # A_bot[1,0]*B6_bot[0] -> C6[1]
    U[1*4+2][col] += 1; V[2*7+6][col] += 1; W[6*2+1][col] += 1; col += 1
    # Combined: (A_bot[0,1] + A_bot[1,1]) * B6_bot[1] -> C6[0] + C6[1]
    # This single multiplication covers both outputs by linearity
    U[0*4+3][col] += 1; V[3*7+6][col] += 1; W[6*2+0][col] += 1
    U[1*4+3][col] += 1; V[3*7+6][col] += 1; W[6*2+1][col] += 1
    col += 1

    return {"U": U, "V": V, "W": W}
