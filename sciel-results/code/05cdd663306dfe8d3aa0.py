"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    rank = 50
    U = [[0] * rank for _ in range(n * m)]   # 8 x 50
    V = [[0] * rank for _ in range(m * p)]   # 28 x 50
    W = [[0] * rank for _ in range(p * n)]   # 14 x 50

    def process_2x2_times_2x7(A_base_u, B_base_v, C_base_w, col_offset):
        """Process 2x2 * 2x7 using 3 Strassen 2x2x2 blocks (cols 0-1, 2-3, 4-5) + 1 schoolbook 2x2x1 (col 6)."""
        a_u = A_base_u + 0
        b_u = A_base_u + 1
        c_u = A_base_u + 2
        d_u = A_base_u + 3

        # Three Strassen 2x2x2 blocks for column pairs (0,1), (2,3), (4,5)
        for group in range(3):
            col_start = group * 2
            e_v = B_base_v + col_start
            f_v = B_base_v + col_start + 1
            g_v = B_base_v + 7 + col_start
            h_v = B_base_v + 7 + col_start + 1

            c00_w = C_base_w + col_start
            c01_w = C_base_w + col_start + 1
            c10_w = C_base_w + 7 + col_start
            c11_w = C_base_w + 7 + col_start + 1

            # M1 = (a+d)(e+h)
            U[a_u][col_offset] = 1
            U[d_u][col_offset] = 1
            V[e_v][col_offset] = 1
            V[h_v][col_offset] = 1
            W[c00_w][col_offset] = 1
            W[c11_w][col_offset] = 1
            col_offset += 1

            # M2 = (c+d)e
            U[c_u][col_offset] = 1
            U[d_u][col_offset] = 1
            V[e_v][col_offset] = 1
            W[c10_w][col_offset] = 1
            W[c11_w][col_offset] = -1
            col_offset += 1

            # M3 = a(f-h)
            U[a_u][col_offset] = 1
            V[f_v][col_offset] = 1
            V[h_v][col_offset] = -1
            W[c01_w][col_offset] = 1
            W[c11_w][col_offset] = 1
            col_offset += 1

            # M4 = d(g-e)
            U[d_u][col_offset] = 1
            V[g_v][col_offset] = 1
            V[e_v][col_offset] = -1
            W[c00_w][col_offset] = 1
            W[c10_w][col_offset] = 1
            col_offset += 1

            # M5 = (a+b)h
            U[a_u][col_offset] = 1
            U[b_u][col_offset] = 1
            V[h_v][col_offset] = 1
            W[c00_w][col_offset] = -1
            W[c01_w][col_offset] = 1
            col_offset += 1

            # M6 = (c-a)(e+f)
            U[c_u][col_offset] = 1
            U[a_u][col_offset] = -1
            V[e_v][col_offset] = 1
            V[f_v][col_offset] = 1
            W[c11_w][col_offset] = 1
            col_offset += 1

            # M7 = (b-d)(g+h)
            U[b_u][col_offset] = 1
            U[d_u][col_offset] = -1
            V[g_v][col_offset] = 1
            V[h_v][col_offset] = 1
            W[c00_w][col_offset] = 1
            col_offset += 1

        # Schoolbook 2x2x1 for column 6
        e_v = B_base_v + 6
        g_v = B_base_v + 13
        c0_w = C_base_w + 6
        c1_w = C_base_w + 13

        U[a_u][col_offset] = 1
        V[e_v][col_offset] = 1
        W[c0_w][col_offset] = 1
        col_offset += 1

        U[b_u][col_offset] = 1
        V[g_v][col_offset] = 1
        W[c0_w][col_offset] = 1
        col_offset += 1

        U[c_u][col_offset] = 1
        V[e_v][col_offset] = 1
        W[c1_w][col_offset] = 1
        col_offset += 1

        U[d_u][col_offset] = 1
        V[g_v][col_offset] = 1
        W[c1_w][col_offset] = 1
        col_offset += 1

        return col_offset

    # A_L (U rows 0-3) * B_T (V rows 0-13) -> C (W rows 0-13)
    col_offset = process_2x2_times_2x7(0, 0, 0, 0)
    # A_R (U rows 4-7) * B_B (V rows 14-27) -> C (W rows 0-13)
    col_offset = process_2x2_times_2x7(4, 14, 0, col_offset)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
