import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Rank-56 decomposition via 8 independent 2x2*2x2 Strassen blocks
    # A is 2x4, B is 4x8, C is 2x8
    # Split into 8 blocks: 2 (A column pairs) x 4 (B row pairs) x 4 (C column pairs)
    # Wait: A has 2 rows, 4 cols. B has 4 rows, 8 cols.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can split A's 4 columns into 2 groups of 2: A[:, 0:2] and A[:, 2:4]
    # B's 4 rows into 2 groups of 2: B[0:2, :] and B[2:4, :]
    # B's 8 columns into 4 groups of 2: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8]
    # So we have 2*2 = 4 block multiplications of 2x2 * 2x8
    # Each 2x2 * 2x8 can be split into 4 2x2 * 2x2 multiplications
    # Total: 4 * 4 = 16 blocks of 2x2 * 2x2
    # Each has rank 7, total rank = 16 * 7 = 112? No wait.
    
    # Let me re-examine. A is 2x4, B is 4x8.
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # Split j into two groups: j in {0,1} and j in {2,3}
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    # First term: (A[:, 0:2]) @ (B[0:2, :]) -> 2x2 @ 2x8 = 2x8
    # Second term: (A[:, 2:4]) @ (B[2:4, :]) -> 2x2 @ 2x8 = 2x8
    # So C = A_L @ B_T + A_R @ B_B
    # A_L is 2x2, B_T is 2x8. A_R is 2x2, B_B is 2x8.
    # To compute A_L @ B_T: split B_T into 4 2x2 blocks: B_T = [B0, B1, B2, B3]
    # A_L @ B_T = [A_L@B0, A_L@B1, A_L@B2, A_L@B3]
    # Each A_L@B_i is 2x2 * 2x2, rank 7. Total for A_L@B_T: 4*7 = 28.
    # Similarly for A_R@B_B: 4*7 = 28.
    # Total rank = 56.
    
    # Strassen 2x2*2x2 factors
    # A indices: 0=a00, 1=a01, 2=a10, 3=a11
    # B indices: 0=b00, 1=b01, 2=b10, 3=b11
    # C indices: 0=c00, 1=c01, 2=c10, 3=c11
    U_loc = [[0]*7 for _ in range(4)]
    V_loc = [[0]*7 for _ in range(4)]
    W_loc = [[0]*7 for _ in range(4)]

    # M1 = (a00+a11)(b00+b11) -> c00+1, c11+1
    U_loc[0][0]=1; U_loc[3][0]=1
    V_loc[0][0]=1; V_loc[3][0]=1
    W_loc[0][0]=1; W_loc[3][0]=1
    # M2 = (a10+a11)b00 -> c10+1, c11-1
    U_loc[2][1]=1; U_loc[3][1]=1
    V_loc[0][1]=1
    W_loc[2][1]=1; W_loc[3][1]=-1
    # M3 = a00(b01-b11) -> c01+1, c11+1
    U_loc[0][2]=1
    V_loc[1][2]=1; V_loc[3][2]=-1
    W_loc[1][2]=1; W_loc[3][2]=1
    # M4 = a11(b10-b00) -> c00+1, c10+1
    U_loc[3][3]=1
    V_loc[2][3]=1; V_loc[0][3]=-1
    W_loc[0][3]=1; W_loc[2][3]=1
    # M5 = (a00+a01)b11 -> c00-1, c01+1
    U_loc[0][4]=1; U_loc[1][4]=1
    V_loc[3][4]=1
    W_loc[0][4]=-1; W_loc[1][4]=1
    # M6 = (a10-a00)(b00+b01) -> c11+1
    U_loc[2][5]=1; U_loc[0][5]=-1
    V_loc[0][5]=1; V_loc[1][5]=1
    W_loc[3][5]=1
    # M7 = (a01-a11)(b10+b11) -> c00+1
    U_loc[1][6]=1; U_loc[3][6]=-1
    V_loc[2][6]=1; V_loc[3][6]=1
    W_loc[0][6]=1

    total_rank = 56
    U = [[0]*total_rank for _ in range(n*m)]   # 8 x 56
    V = [[0]*total_rank for _ in range(m*p)]   # 32 x 56
    W = [[0]*total_rank for _ in range(p*n)]   # 16 x 56

    # 4 groups: (a_half, b_row_half) in {(0,0), (0,1), (1,0), (1,1)}
    # For each group, 4 column blocks of C (jj in 0..3)
    # Total 16 blocks, each rank 7, total 112? No, 4*4*7 = 112. That's wrong.
    # Wait: 2 a_halves * 2 b_row_halves * 4 col_blocks = 16 blocks * 7 = 112.
    # But I said 56 above. Let me recheck.
    # A_L @ B_T: A_L is 2x2 (from A[:, 0:2]), B_T is 2x8 (from B[0:2, :])
    # B_T has 2 rows, 8 cols. Split into 4 blocks of 2x2.
    # So A_L @ B_T = 4 independent 2x2*2x2 multiplications, each rank 7. Total 28.
    # A_R @ B_B: similarly 28.
    # Total = 56.
    # So the groups are:
    # Group 0: A[:, 0:2] @ B[0:2, :] -> 4 col blocks
    # Group 1: A[:, 2:4] @ B[0:2, :] -> 4 col blocks? No!
    # A_R = A[:, 2:4], B_B = B[2:4, :]
    # So:
    # Group 0: A[:, 0:2] @ B[0:2, :]
    # Group 1: A[:, 2:4] @ B[2:4, :]
    # Each group has 4 col blocks.
    # Total 2 groups * 4 col blocks * 7 = 56. Correct.

    group_idx = 0
    for a_half in range(2):  # 0: A[:, 0:2], 1: A[:, 2:4]
        b_row_half = a_half  # 0: B[0:2, :], 1: B[2:4, :]
        for jj in range(4):  # 4 column blocks of C
            base_r = group_idx * 7
            group_idx += 1

            # A indices for this half
            # A is 2x4, row-major. A[i, j] -> i*m + j = i*4 + j
            # a00 -> (0, 2*a_half), index = 0*4 + 2*a_half
            # a01 -> (0, 2*a_half+1), index = 0*4 + 2*a_half+1
            # a10 -> (1, 2*a_half), index = 1*4 + 2*a_half
            # a11 -> (1, 2*a_half+1), index = 1*4 + 2*a_half+1
            a_map = [
                0*m + (2*a_half + 0),
                0*m + (2*a_half + 1),
                1*m + (2*a_half + 0),
                1*m + (2*a_half + 1),
            ]

            # B indices: B is 4x8, row-major. B[j, k] -> j*p + k = j*8 + k
            # b00 -> (2*b_row_half, 2*jj), index = (2*b_row_half)*8 + 2*jj
            # b01 -> (2*b_row_half, 2*jj+1), index = (2*b_row_half)*8 + 2*jj+1
            # b10 -> (2*b_row_half+1, 2*jj), index = (2*b_row_half+1)*8 + 2*jj
            # b11 -> (2*b_row_half+1, 2*jj+1), index = (2*b_row_half+1)*8 + 2*jj+1
            b_map = [
                (2*b_row_half + 0)*p + (2*jj + 0),
                (2*b_row_half + 0)*p + (2*jj + 1),
                (2*b_row_half + 1)*p + (2*jj + 0),
                (2*b_row_half + 1)*p + (2*jj + 1),
            ]

            # C indices: C is 2x8, trace convention W[k*n+i, r]
            # C[i, k] -> k*n + i
            # c00 -> (0, 2*jj), index = (2*jj)*2 + 0 = 4*jj
            # c01 -> (0, 2*jj+1), index = (2*jj+1)*2 + 0 = 4*jj+2
            # c10 -> (1, 2*jj), index = (2*jj)*2 + 1 = 4*jj+1
            # c11 -> (1, 2*jj+1), index = (2*jj+1)*2 + 1 = 4*jj+3
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

    if time.time() - start_time > 120:
        pass

    return {"U": U, "V": V, "W": W}
