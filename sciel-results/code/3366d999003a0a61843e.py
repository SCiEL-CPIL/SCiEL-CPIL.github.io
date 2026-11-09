"""Matrix multiplication 2x3 x 3x9 with rank-51 decomposition via tiled 2x3x3 blocks."""
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

    def get_2x3x3_block():
        """Rank-17 decomposition for 2x3 x 3x3.
        A (2x3): a00,a01,a02 / a10,a11,a12  -> U indices 0-5
        B (3x3): b00,b01,b02 / b10,b11,b12 / b20,b21,b22  -> V indices 0-8
        C (2x3): c00,c01,c02 / c10,c11,c12  -> W indices 0-5 (k*n+i)
        """
        r = 17
        U = [[0] * r for _ in range(6)]
        V = [[0] * r for _ in range(9)]
        W = [[0] * r for _ in range(6)]

        # Laderman for top-left 2x2 x 2x2 (A rows 0,1 cols 0,1; B rows 0,1 cols 0,1)
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
        # U: 4=a02, 5=a12; V: 6=b20, 7=b21; W: 0=c00,1=c01,2=c10,3=c11
        U[4][7] = 1; V[6][7] = 1; W[0][7] = 1
        U[4][8] = 1; V[7][8] = 1; W[1][8] = 1
        U[5][9] = 1; V[6][9] = 1; W[2][9] = 1
        U[5][10] = 1; V[7][10] = 1; W[3][10] = 1

        # Part 3: a00*b02, a01*b12, a10*b02, a11*b12
        # U: 0,1,2,3; V: 4=b02, 5=b12; W: 4=c02, 5=c12
        U[0][11] = 1; V[4][11] = 1; W[4][11] = 1
        U[1][12] = 1; V[5][12] = 1; W[4][12] = 1
        U[2][13] = 1; V[4][13] = 1; W[5][13] = 1
        U[3][14] = 1; V[5][14] = 1; W[5][14] = 1

        # Part 4: a02*b22, a12*b22
        # U: 4,5; V: 8=b22; W: 4=c02, 5=c12
        U[4][15] = 1; V[8][15] = 1; W[4][15] = 1
        U[5][16] = 1; V[8][16] = 1; W[5][16] = 1

        return U, V, W

    def verify_full(U, V, W, n, m, p):
        np.random.seed(12345)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B

        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        r = len(U_np[0])

        C_actual = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for t in range(r):
                        s += U_np[i * m + j, t] * V_np[j * p + k, t] * W_np[k * n + i, t]
                C_actual[i, k] = s

        return np.allclose(C_actual, C_expected, atol=1e-9)

    # Build and verify the block
    U_block, V_block, W_block = get_2x3x3_block()

    # Verify the block independently
    block_ok = verify_full(U_block, V_block, W_block, 2, 3, 3)

    if block_ok:
        # Tile 3 blocks for 2x3 x 3x9
        r_total = 51
        U = [[0] * r_total for _ in range(6)]
        V = [[0] * r_total for _ in range(27)]
        W = [[0] * r_total for _ in range(18)]

        for b in range(3):
            offset = b * 17
            # U: same 6 rows for each block
            for i in range(6):
                for t in range(17):
                    U[i][offset + t] = U_block[i][t]

            # V: block b uses V rows j*3+k for j=0..2, k=0..2
            # Global V row = j*9 + (b*3 + k)
            for j in range(3):
                for k in range(3):
                    for t in range(17):
                        V[j * 9 + (b * 3 + k)][offset + t] = V_block[j * 3 + k][t]

            # W: block b uses W rows k*2+i for k=0..2, i=0..1
            # Global W row = (b*3 + k)*2 + i
            for k in range(3):
                for i in range(2):
                    for t in range(17):
                        W[(b * 3 + k) * 2 + i][offset + t] = W_block[k * 2 + i][t]

        if verify_full(U, V, W, 2, 3, 9):
            return {"U": U, "V": V, "W": W}

    if time.time() > deadline - 5.0:
        return schoolbook()

    return schoolbook()


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
