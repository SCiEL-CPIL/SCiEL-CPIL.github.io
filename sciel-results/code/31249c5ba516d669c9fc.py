import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    DEADLINE = 110.0

    n, m, p = 2, 4, 8

    # Rank 56 decomposition using 8 independent Strassen 2x2x2 blocks
    # Structure:
    #   A (2x4) = [A_L (2x2) | A_R (2x2)]
    #   B (4x8) = [[B_00 (2x4) | B_01 (2x4)]]  (top 2 rows, split into two 2x2 blocks)
    #              [[B_10 (2x4) | B_11 (2x4)]]  (bottom 2 rows, split into two 2x2 blocks)
    #   C = A_L * B_top + A_R * B_bottom
    #
    # Each 2x2 * 2x4 product uses 2 Strassen 2x2x2 multiplications (for left and right 2x2 halves of B)
    # 4 (A_L/A_R) * 2 (B_top/B_bottom) * 2 (left/right halves) = 16 ... wait
    # Actually: A_L * B_00 where B_00 is 2x4 = A_L * [B_00L | B_00R]
    #   = [A_L*B_00L | A_L*B_00R]
    #   Each A_L*B_00L and A_L*B_00R is a 2x2*2x2 Strassen (rank 7)
    # So A_L * B_00 uses 2 Strassen (rank 14)
    # Similarly A_L * B_01 uses 2 Strassen (rank 14)
    # A_R * B_10 uses 2 Strassen (rank 14)
    # A_R * B_11 uses 2 Strassen (rank 14)
    # Total: 4 * 2 * 7 = 56

    rank = 56
    U = [[0] * rank for _ in range(n * m)]   # (8, 56)
    V = [[0] * rank for _ in range(m * p)]   # (32, 56)
    W = [[0] * rank for _ in range(p * n)]   # (16, 56)

    # Strassen for 2x2 * 2x2:
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    # P0 = (a0+a3)(b0+b3)
    # P1 = (a2+a3)b0
    # P2 = a0(b1-b2)
    # P3 = a3(b2+b3)
    # P4 = (a0+a1)b2
    # P5 = (a1-a2)(b0+b1)
    # P6 = (a3-a2)(b3-b2)
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5

    # X_forms: for each Strassen product s, the coefficients of A elements [a0, a1, a2, a3]
    X_forms = [
        [1, 0, 0, 1],   # P0: a0 + a3
        [0, 0, 1, 1],   # P1: a2 + a3
        [1, 0, 0, 0],   # P2: a0
        [0, 0, 0, 1],   # P3: a3
        [1, 1, 0, 0],   # P4: a0 + a1
        [-1, 0, 1, 0],  # P5: -a0 + a2
        [0, 1, 0, -1]   # P6: a1 - a3
    ]

    # Y_forms: for each Strassen product s, the coefficients of B elements [b0, b1, b2, b3]
    Y_forms = [
        [1, 0, 0, 1],   # P0: b0 + b3
        [1, 0, 0, 0],   # P1: b0
        [0, 1, 0, -1],  # P2: b1 - b2
        [-1, 0, 1, 0],  # P3: -b0 + b2
        [0, 0, 0, 1],   # P4: b2
        [1, 1, 0, 0],   # P5: b0 + b1
        [0, 0, 1, 1]    # P6: b2 + b3
    ]

    # Z_combos: for each C element (c00, c01, c10, c11), the coefficients of [P0, P1, P2, P3, P4, P5, P6]
    Z_combos = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ]

    # 8 Strassen instances:
    # For each of the 4 (a_block, b_block) pairs:
    #   a_block in {0, 1}: A_L (cols 0-1) or A_R (cols 2-3)
    #   b_block in {0, 1, 2, 3}: B rows 0-1 cols 0-1, B rows 0-1 cols 2-3, B rows 2-3 cols 0-1, B rows 2-3 cols 2-3
    # Actually, let me think about this differently.
    #
    # A is 2x4. Split into A_L (cols 0-1) and A_R (cols 2-3).
    # B is 4x8. Split into 4 blocks of 2x2:
    #   B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
    #   B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6], B[2:4, 6:8]
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #        = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    #        = (A_L * B_top)[i,k] + (A_R * B_bottom)[i,k]
    #
    # Where B_top is B[0:2, :] (2x8) and B_bottom is B[2:4, :] (2x8)
    #
    # A_L * B_top: 2x2 * 2x8 = 2x8
    #   Split B_top into 4 blocks of 2x2: B_top[:, 0:2], B_top[:, 2:4], B_top[:, 4:6], B_top[:, 6:8]
    #   Each A_L * B_top[:, c:c+2] is a 2x2 * 2x2 Strassen (rank 7)
    #   Total for A_L * B_top: 4 * 7 = 28
    #
    # Similarly A_R * B_bottom: 2x2 * 2x8 = 2x8
    #   Total: 4 * 7 = 28
    #
    # Grand total: 56

    # Define the 8 Strassen instances (actually 16, but let me recount)
    # Wait, 4 blocks for A_L * B_top + 4 blocks for A_R * B_bottom = 8 instances
    # Each instance is one Strassen 2x2x2 (rank 7)
    # 8 * 7 = 56. Correct.

    # Instance (a_block, b_col_block):
    #   a_block=0: A_L (A cols 0-1), a_block=1: A_R (A cols 2-3)
    #   b_col_block in {0,1,2,3}: B col offset = b_col_block * 2
    #   b_row_block: for a_block=0, b rows 0-1; for a_block=1, b rows 2-3

    instances = []
    for a_block in range(2):
        for b_col_block in range(4):
            # a_col_off: column offset in A
            a_col_off = a_block * 2
            # b_row_off: row offset in B
            b_row_off = a_block * 2  # a_block=0 -> rows 0-1, a_block=1 -> rows 2-3
            # b_col_off: column offset in B
            b_col_off = b_col_block * 2
            instances.append((a_col_off, b_row_off, b_col_off))

    r_idx = 0
    for (a_col_off, b_row_off, b_col_off) in instances:
        for s in range(7):
            # Fill U: A[i, j] where i in [0,2), j = a_col_off + jp
            for i in range(n):
                for jp in range(2):
                    a_idx = i * m + a_col_off + jp
                    U[a_idx][r_idx] = X_forms[s][i * 2 + jp]

            # Fill V: B[j, k] where j = b_row_off + jp, k = b_col_off + kp
            for jp in range(2):
                for kp in range(2):
                    b_idx = (b_row_off + jp) * p + (b_col_off + kp)
                    V[b_idx][r_idx] = Y_forms[s][jp * 2 + kp]

            # Fill W: C[k, i] where k = b_col_off + kp, i in [0,2)
            for i in range(n):
                for kp in range(2):
                    k = b_col_off + kp
                    c_idx = k * n + i
                    W[c_idx][r_idx] = Z_combos[i * 2 + kp][s]

            r_idx += 1

    return {"U": U, "V": V, "W": W}

if __name__ == '__main__':
    res = construct()
    U = res['U']
    V = res['V']
    W = res['W']
    
    n, m, p = 2, 4, 8
    assert len(U) == n*m, "U shape error"
    assert len(U[0]) == len(V[0]), "Rank mismatch U V"
    assert len(V) == m*p, "V shape error"
    assert len(W) == p*n, "W shape error"
    assert len(W[0]) == len(U[0]), "Rank mismatch W U"
    
    import random
    random.seed(42)
    A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
    
    C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    C_prime = [[0] * p for _ in range(n)]
    for r in range(len(U[0])):
        u_val = sum(U[i*m + j][r] * A[i][j] for i in range(n) for j in range(m))
        v_val = sum(V[j*p + k][r] * B[j][k] for j in range(m) for k in range(p))
        prod = u_val * v_val
        for k in range(p):
            for i in range(n):
                C_prime[i][k] += W[k*n + i][r] * prod
    
    for i in range(n):
        for k in range(p):
            if C[i][k] != C_prime[i][k]:
                print(f"Mismatch at C[{i}][{k}]: {C[i][k]} vs {C_prime[i][k]}")
                import sys
                sys.exit(1)
    
    print(f"Verified correct. Rank = {len(U[0])}")
