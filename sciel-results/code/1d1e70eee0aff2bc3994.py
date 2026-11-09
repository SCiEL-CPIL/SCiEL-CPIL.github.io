"""Matrix multiplication 2x3 x 3x9 with rank-minimal decomposition via tiled 2x3x3 blocks."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 2, 3, 9
    rank_nm = n * m * p   # 54

    def schoolbook():
        r = rank_nm
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    def get_2x3x3_block_rank17():
        """Rank-17 decomposition for 2x3 x 3x3."""
        r = 17
        U = [[0] * r for _ in range(6)]
        V = [[0] * r for _ in range(9)]
        W = [[0] * r for _ in range(6)]

        # Laderman for top-left 2x2 x 2x2
        # U: 0=a00, 1=a01, 2=a10, 3=a11
        # V: 0=b00, 1=b01, 2=b10, 3=b11
        # W: 0=c00, 1=c01, 2=c10, 3=c11

        # S1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1

        # S2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1

        # S3 = a00(b10-b11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1

        # S4 = a11(b01-b00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = -1; W[1][3] = 1

        # S5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1; W[3][4] = -1

        # S6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1

        # S7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = -1; W[2][6] = 1

        # Part 2: a02*b20, a02*b21, a12*b20, a12*b21
        U[4][7] = 1; V[6][7] = 1; W[0][7] = 1
        U[4][8] = 1; V[7][8] = 1; W[1][8] = 1
        U[5][9] = 1; V[6][9] = 1; W[2][9] = 1
        U[5][10] = 1; V[7][10] = 1; W[3][10] = 1

        # Part 3: a00*b02, a01*b12, a10*b02, a11*b12
        U[0][11] = 1; V[4][11] = 1; W[4][11] = 1
        U[1][12] = 1; V[5][12] = 1; W[4][12] = 1
        U[2][13] = 1; V[4][13] = 1; W[5][13] = 1
        U[3][14] = 1; V[5][14] = 1; W[5][14] = 1

        # Part 4: a02*b22, a12*b22
        U[4][15] = 1; V[8][15] = 1; W[4][15] = 1
        U[5][16] = 1; V[8][16] = 1; W[5][16] = 1

        return U, V, W

    def get_2x3x3_block_rank15():
        """Attempt rank-15 decomposition for 2x3 x 3x3.
        
        Strategy: Use Laderman (7 terms) for the 2x2x2x2 core,
        then handle cross terms involving a02/a12 and b20/b21/b22
        plus b02/b12 with 8 additional terms.
        """
        r = 15
        U = [[0] * r for _ in range(6)]
        V = [[0] * r for _ in range(9)]
        W = [[0] * r for _ in range(6)]

        # Laderman for 2x2x2x2 (indices 0-6)
        # U: 0=a00, 1=a01, 2=a10, 3=a11
        # V: 0=b00, 1=b01, 2=b10, 3=b11
        # W: 0=c00, 1=c01, 2=c10, 3=c11

        # S1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1

        # S2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1

        # S3 = a00(b10-b11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1

        # S4 = a11(b01-b00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = -1; W[1][3] = 1

        # S5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = 1; W[3][4] = -1

        # S6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1

        # S7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = -1; W[2][6] = 1

        # Now handle: a02*b20->c00, a02*b21->c01, a12*b20->c10, a12*b21->c11
        # a00*b02->c00, a01*b12->c00 (wait, no: a[i,j]*b[j,k])
        # Let me be precise:
        # c00 = a00*b00 + a01*b10 + a02*b20
        # c01 = a00*b01 + a01*b11 + a02*b21
        # c02 = a00*b02 + a01*b12 + a02*b22
        # c10 = a10*b00 + a11*b10 + a12*b20
        # c11 = a10*b01 + a11*b11 + a12*b21
        # c12 = a10*b02 + a11*b12 + a12*b22

        # Laderman handles all terms with a in {a00,a01,a10,a11} and b in {b00,b01,b10,b11}
        # Remaining terms:
        # c00: + a02*b20
        # c01: + a02*b21
        # c02: a00*b02 + a01*b12 + a02*b22
        # c10: + a12*b20
        # c11: + a12*b21
        # c12: a10*b02 + a11*b12 + a12*b22

        # 8 remaining scalar products. Can we do better than 8?
        # Try grouping:
        # T8 = a02*(b20 + b21): contributes to c00 and c01
        # But we need separate outputs...
        # T8 = a02*b20 -> c00 (1 term)
        # T9 = a02*b21 -> c01 (1 term)
        # T10 = a12*b20 -> c10 (1 term)
        # T11 = a12*b21 -> c11 (1 term)
        # For c02 and c12:
        # c02 = a00*b02 + a01*b12 + a02*b22
        # c12 = a10*b02 + a11*b12 + a12*b22
        # T12 = b02*(a00+a10) -> c02 + c12
        # T13 = b12*(a01+a11) -> c02 + c12
        # T14 = a02*b22 -> c02
        # T15 = a12*b22 -> c12
        # That's 8 terms: T8-T15. Total = 7 + 8 = 15.

        # T8: a02*b20 -> c00
        U[4][7] = 1; V[6][7] = 1; W[0][7] = 1

        # T9: a02*b21 -> c01
        U[4][8] = 1; V[7][8] = 1; W[1][8] = 1

        # T10: a12*b20 -> c10
        U[5][9] = 1; V[6][9] = 1; W[2][9] = 1

        # T11: a12*b21 -> c11
        U[5][10] = 1; V[7][10] = 1; W[3][10] = 1

        # T12: (a00+a10)*b02 -> c02 + c12
        U[0][11] = 1; U[2][11] = 1
        V[4][11] = 1
        W[4][11] = 1; W[5][11] = 1

        # T13: (a01+a11)*b12 -> c02 + c12
        U[1][12] = 1; U[3][12] = 1
        V[5][12] = 1
        W[4][12] = 1; W[5][12] = 1

        # T14: a02*b22 -> c02
        U[4][13] = 1; V[8][13] = 1; W[4][13] = 1

        # T15: a12*b22 -> c12
        U[5][14] = 1; V[8][14] = 1; W[5][14] = 1

        return U, V, W

    def verify_full(U, V, W, n, m, p, seed=12345):
        np.random.seed(seed)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B

        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        r = len(U_np[0])

        C_actual = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for t in range(r):
                        s += U_np[i * m + j, t] * V_np[j * p + k, t] * W_np[k * n + i, t]
                C_actual[i, k] = s

        return np.allclose(C_actual, C_expected, atol=1e-9), C_actual, C_expected

    def tile_blocks(U_block, V_block, W_block, r_block, n=2, m=3, p=9):
        """Tile 3 blocks of size 2x3x3 to form 2x3x9."""
        r_total = r_block * 3
        U = [[0] * r_total for _ in range(n * m)]
        V = [[0] * r_total for _ in range(m * p)]
        W = [[0] * r_total for _ in range(p * n)]

        for b in range(3):
            offset = b * r_block
            for i in range(n * m):
                for t in range(r_block):
                    U[i][offset + t] = U_block[i][t]

            for j in range(3):
                for k in range(3):
                    for t in range(r_block):
                        V[j * 9 + (b * 3 + k)][offset + t] = V_block[j * 3 + k][t]

            for k in range(3):
                for i in range(2):
                    for t in range(r_block):
                        W[(b * 3 + k) * 2 + i][offset + t] = W_block[k * 2 + i][t]

        return U, V, W

    # Try rank-15 block first
    if time.time() < deadline - 10.0:
        U15, V15, W15 = get_2x3x3_block_rank15()
        ok15, _, _ = verify_full(U15, V15, W15, 2, 3, 3)

        if ok15:
            U_full, V_full, W_full = tile_blocks(U15, V15, W15, 15)
            ok_full, _, _ = verify_full(U_full, V_full, W_full, 2, 3, 9)
            if ok_full:
                return {"U": U_full, "V": V_full, "W": W_full}

    # Try rank-17 block
    if time.time() < deadline - 5.0:
        U17, V17, W17 = get_2x3x3_block_rank17()
        ok17, _, _ = verify_full(U17, V17, W17, 2, 3, 3)

        if ok17:
            U_full, V_full, W_full = tile_blocks(U17, V17, W17, 17)
            ok_full, _, _ = verify_full(U_full, V_full, W_full, 2, 3, 9)
            if ok_full:
                return {"U": U_full, "V": V_full, "W": W_full}

    return schoolbook()


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
