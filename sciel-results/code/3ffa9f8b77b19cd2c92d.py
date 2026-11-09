"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    n, m, p = DIMENSIONS
    # Rank 66 decomposition
    # 2 blocks of A (3x2), 2 blocks of B (2x6)
    # Each 3x2x6 is split into 3 3x2x2 blocks
    # Each 3x2x2 has rank 11
    # Total rank = 2 * 3 * 11 = 66
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct 3x2x2 sub-decomposition
    # U_s: 6x11
    # V_s: 4x11
    # W_s: 6x11
    
    U_s = [[0]*11 for _ in range(6)]
    V_s = [[0]*11 for _ in range(4)]
    W_s = [[0]*11 for _ in range(6)]
    
    # M1
    U_s[0][0] += 1; U_s[3][0] += 1
    V_s[0][0] += 1; V_s[3][0] += 1
    W_s[0][0] += 1; W_s[3][0] += 1
    
    # M2
    U_s[2][1] += 1; U_s[3][1] += 1
    V_s[0][1] += 1
    W_s[2][1] += 1; W_s[3][1] -= 1
    
    # M3
    U_s[0][2] += 1
    V_s[1][2] += 1; V_s[3][2] -= 1
    W_s[1][2] += 1; W_s[3][2] += 1
    
    # M4
    U_s[3][3] += 1
    V_s[2][3] += 1; V_s[0][3] -= 1
    W_s[0][3] += 1; W_s[2][3] += 1
    
    # M5
    U_s[0][4] += 1; U_s[1][4] += 1
    V_s[3][4] += 1
    W_s[0][4] -= 1; W_s[1][4] += 1
    
    # M6
    U_s[2][5] += 1; U_s[0][5] -= 1
    V_s[0][5] += 1; V_s[1][5] += 1
    W_s[3][5] += 1
    
    # M7
    U_s[1][6] += 1; U_s[3][6] -= 1
    V_s[2][6] += 1; V_s[3][6] += 1
    W_s[0][6] += 1
    
    # M8
    U_s[4][7] += 1
    V_s[0][7] += 1
    W_s[4][7] += 1
    
    # M9
    U_s[5][8] += 1
    V_s[2][8] += 1
    W_s[4][8] += 1
    
    # M10
    U_s[4][9] += 1
    V_s[1][9] += 1
    W_s[5][9] += 1
    
    # M11
    U_s[5][10] += 1
    V_s[3][10] += 1
    W_s[5][10] += 1

    # Assemble Global U, V, W
    
    # A_L B_T (cols 0-32)
    for b in range(3):
        for k in range(11):
            c = b * 11 + k
            # U_s to U
            for rs in range(6):
                if U_s[rs][k] != 0:
                    i = rs // 2
                    jp = rs % 2
                    rg = i * 4 + jp
                    U[rg][c] += U_s[rs][k]
            # V_s to V
            for rs in range(4):
                if V_s[rs][k] != 0:
                    jp = rs // 2
                    kp = rs % 2
                    rg = jp * 6 + (2 * b + kp)
                    V[rg][c] += V_s[rs][k]
            # W_s to W
            for rs in range(6):
                if W_s[rs][k] != 0:
                    kp = rs // 3
                    i = rs % 3
                    rg = (2 * b + kp) * 3 + i
                    W[rg][c] += W_s[rs][k]

    # A_R B_B (cols 33-65)
    for b in range(3):
        for k in range(11):
            c = 33 + b * 11 + k
            # U_s to U
            for rs in range(6):
                if U_s[rs][k] != 0:
                    i = rs // 2
                    jp = rs % 2
                    rg = i * 4 + (2 + jp)
                    U[rg][c] += U_s[rs][k]
            # V_s to V
            for rs in range(4):
                if V_s[rs][k] != 0:
                    jp = rs // 2
                    kp = rs % 2
                    rg = (2 + jp) * 6 + (2 * b + kp)
                    V[rg][c] += V_s[rs][k]
            # W_s to W
            for rs in range(6):
                if W_s[rs][k] != 0:
                    kp = rs // 3
                    i = rs % 3
                    rg = (2 * b + kp) * 3 + i
                    W[rg][c] += W_s[rs][k]

    if time.time() > deadline:
        # Fallback to schoolbook if somehow slow (should not happen)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
