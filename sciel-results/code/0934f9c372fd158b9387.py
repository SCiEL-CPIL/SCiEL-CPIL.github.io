import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Use Strassen for 2x2 sub-matrices.
    # A is 2x4, B is 4x8.
    # Split A into two 2x2 blocks horizontally: A = [A_L, A_R]
    # Split B into two 2x8 blocks vertically: B = [B_T; B_B]
    # C = A_L @ B_T + A_R @ B_B
    # A_L @ B_T is 2x2 @ 2x8.
    # A_R @ B_B is 2x2 @ 2x8.
    # To multiply 2x2 by 2x8, we can split B into four 2x2 blocks horizontally.
    # Let B_T = [B_0, B_1, B_2, B_3] where each B_i is 2x2.
    # Then A_L @ B_T = [A_L@B_0, A_L@B_1, A_L@B_2, A_L@B_3].
    # Each A_L@B_i is a 2x2 matrix multiply, rank 7.
    # So A_L @ B_T costs 4 * 7 = 28 rank.
    # Similarly A_R @ B_B costs 4 * 7 = 28 rank.
    # Total rank = 56.

    # Local Strassen factors for 2x2 * 2x2 -> 2x2
    # Indices for A: 0=00, 1=01, 2=10, 3=11
    # Indices for B: 0=00, 1=01, 2=10, 3=11
    # Indices for C: 0=00, 1=01, 2=10, 3=11
    U_loc = [[0]*7 for _ in range(4)]
    V_loc = [[0]*7 for _ in range(4)]
    W_loc = [[0]*7 for _ in range(4)]

    # r=0: M1=(a00+a11)(b00+b11) -> c00+1, c11+1
    U_loc[0][0]=1; U_loc[3][0]=1
    V_loc[0][0]=1; V_loc[3][0]=1
    W_loc[0][0]=1; W_loc[3][0]=1
    # r=1: M2=(a10+a11)b00 -> c10+1, c11-1
    U_loc[2][1]=1; U_loc[3][1]=1
    V_loc[0][1]=1
    W_loc[2][1]=1; W_loc[3][1]=-1
    # r=2: M3=a00(b01-b11) -> c01+1, c11+1
    U_loc[0][2]=1
    V_loc[1][2]=1; V_loc[3][2]=-1
    W_loc[1][2]=1; W_loc[3][2]=1
    # r=3: M4=a11(b10-b00) -> c00+1, c10+1
    U_loc[3][3]=1
    V_loc[2][3]=1; V_loc[0][3]=-1
    W_loc[0][3]=1; W_loc[2][3]=1
    # r=4: M5=(a00+a01)b11 -> c00-1, c01+1
    U_loc[0][4]=1; U_loc[1][4]=1
    V_loc[3][4]=1
    W_loc[0][4]=-1; W_loc[1][4]=1
    # r=5: M6=(a10-a00)(b00+b01) -> c11+1
    U_loc[2][5]=1; U_loc[0][5]=-1
    V_loc[0][5]=1; V_loc[1][5]=1
    W_loc[3][5]=1
    # r=6: M7=(a01-a11)(b10+b11) -> c00+1
    U_loc[1][6]=1; U_loc[3][6]=-1
    V_loc[2][6]=1; V_loc[3][6]=1
    W_loc[0][6]=1

    # Total rank calculation:
    # 2 (A halves) * 4 (B column blocks) * 7 (Strassen rank) = 56
    total_rank = 56

    U = [[0]*total_rank for _ in range(n*m)]   # 8 x 56
    V = [[0]*total_rank for _ in range(m*p)]   # 32 x 56
    W = [[0]*total_rank for _ in range(p*n)]   # 16 x 56

    group = 0
    for half in range(2):  # 0: A[:, 0:2] & B[0:2, :], 1: A[:, 2:4] & B[2:4, :]
        for jj in range(p // 2):  # 4 column blocks of size 2
            base_r = group * 7
            group += 1

            # A half 'half': columns 2*half, 2*half+1
            # A is 2x4, row-major.
            # a00 -> (0, 2h), a01 -> (0, 2h+1)
            # a10 -> (1, 2h), a11 -> (1, 2h+1)
            a_map = [
                0*m + (2*half + 0),
                0*m + (2*half + 1),
                1*m + (2*half + 0),
                1*m + (2*half + 1),
            ]

            # B half 'half': rows 2*half, 2*half+1. Cols 2*jj, 2*jj+1.
            # B is 4x8, row-major.
            # b00 -> (2h, 2j), b01 -> (2h, 2j+1)
            # b10 -> (2h+1, 2j), b11 -> (2h+1, 2j+1)
            b_map = [
                (2*half + 0)*p + (2*jj + 0),
                (2*half + 0)*p + (2*jj + 1),
                (2*half + 1)*p + (2*jj + 0),
                (2*half + 1)*p + (2*jj + 1),
            ]

            # C output for this block: rows 0,1. Cols 2*jj, 2*jj+1.
            # C is 2x8, row-major.
            # c00 -> (0, 2j), c01 -> (0, 2j+1)
            # c10 -> (1, 2j), c11 -> (1, 2j+1)
            # W index = k*n + i (where k is col, i is row)
            c_map = [
                (2*jj + 0)*n + 0,
                (2*jj + 1)*n + 0,
                (2*jj + 0)*n + 1,
                (2*jj + 1)*n + 1,
            ]

            for r in range(7):
                rr = base_r + r
                for la in range(4):
                    val = U_loc[la][r]
                    if val:
                        U[a_map[la]][rr] += val
                for lb in range(4):
                    val = V_loc[lb][r]
                    if val:
                        V[b_map[lb]][rr] += val
                for lc in range(4):
                    val = W_loc[lc][r]
                    if val:
                        W[c_map[lc]][rr] += val

    # Deadline guard
    if time.time() - start_time > 120:
        pass

    return {"U": U, "V": V, "W": W}
