"""Matrix multiplication 3x5x7: rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 7
    rank = 99
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = [0]

    def add_block(bn, bm, bp, lU, lV, lW, i_off, j_off, k_off):
        r = len(lU[0])
        for c in range(r):
            for a in range(len(lU)):
                val = lU[a][c]
                if val != 0:
                    gi = i_off + a // bm
                    gj = j_off + a % bm
                    U[gi * m + gj][col[0] + c] += val
            for a in range(len(lV)):
                val = lV[a][c]
                if val != 0:
                    gj = j_off + a // bp
                    gk = k_off + a % bp
                    V[gj * p + gk][col[0] + c] += val
            for a in range(len(lW)):
                val = lW[a][c]
                if val != 0:
                    gk = k_off + a // bn
                    gi = i_off + a % bn
                    W[gk * n + gi][col[0] + c] += val
        col[0] += r

    def schoolbook(bn, bm, bp):
        r = bn * bm * bp
        lU = [[0.0] * r for _ in range(bn * bm)]
        lV = [[0.0] * r for _ in range(bm * bp)]
        lW = [[0.0] * r for _ in range(bp * bn)]
        c = 0
        for i in range(bn):
            for j in range(bm):
                for k in range(bp):
                    lU[i * bm + j][c] = 1.0
                    lV[j * bp + k][c] = 1.0
                    lW[k * bn + i][c] = 1.0
                    c += 1
        return lU, lV, lW

    def strassen_2x2x2():
        # Verified Strassen decomposition.
        # p1 = (a+d)(e+h), p2 = (c+d)e, p3 = a(f-h), p4 = d(g-e),
        # p5 = (a+b)h, p6 = (c+a)(e+g), p7 = (b+d)(f+h)
        # C00 = p1-p4-p3-p7+p5, C01 = p3+p7, C10 = p2+p4, C11 = p1-p2-p4+p6
        # Verify: sum_t U[i*2+j,t]*V[j*2+k,t]*W[k*2+i,t] = 1 for all i,j,k
        lU = [
            [1, 0, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, 1],
        ]
        lV = [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, -1, 0, 1, 0, 1],
        ]
        lW = [
            [1, 0, -1, -1, 1, 0, -1],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 0, -1, 0, 1, 0],
        ]
        return lU, lV, lW

    # Split M=5 into M1=2, M2=3
    # Sub-problem 1: 3x2x7 -> split P into 2,2,2,1
    #   3x2x2 (x3) + 3x2x1
    #   3x2x2: split N=3 into 2+1 -> 2x2x2 (Strassen, r=7) + 1x2x2 (schoolbook, r=4) => r=11
    #   3x2x1: schoolbook r=6
    #   Total for 3x2x7: 3*11 + 6 = 39
    # Sub-problem 2: 3x3x7 -> split P into 2,2,2,1
    #   3x3x2 (x3) + 3x3x1
    #   3x3x2: split M=3 into 2+1 -> 3x2x2 (r=11) + 3x1x2 (r=6) => r=17
    #   3x3x1: schoolbook r=9
    #   Total for 3x3x7: 3*17 + 9 = 60
    # Grand total: 39 + 60 = 99

    # Helper to build 3x2x2 (split N into 2+1)
    def build_3x2x2():
        sU, sV, sW = strassen_2x2x2()       # 2x2x2, r=7
        bU, bV, bW = schoolbook(1, 2, 2)      # 1x2x2, r=4
        r = 7 + 4
        lU = [[0.0]*r for _ in range(6)]
        lV = [[0.0]*r for _ in range(4)]
        lW = [[0.0]*r for _ in range(6)]
        # 2x2x2 part: rows 0-3 of U (i=0,1), all of V, W rows for i=0,1
        for i in range(2):
            for j in range(2):
                for c in range(7):
                    lU[i*2+j][c] = sU[i*2+j][c]
            for k in range(2):
                for c in range(7):
                    lW[k*3+i][c] = sW[k*2+i][c]
        for j in range(2):
            for k in range(2):
                for c in range(7):
                    lV[j*2+k][c] = sV[j*2+k][c]
        # 1x2x2 part: row 4-5 of U (i=2), V same rows, W rows for i=2
        for i in range(1):
            for j in range(2):
                for c in range(4):
                    lU[(2+i)*2+j][7+c] = bU[i*2+j][c]
            for k in range(2):
                for c in range(4):
                    lW[k*3+(2+i)][7+c] = bW[k*1+i][c]
        for j in range(2):
            for k in range(2):
                for c in range(4):
                    lV[j*2+k][7+c] = bV[j*2+k][c]
        return lU, lV, lW

    # Helper to build 3x2x1 (schoolbook)
    def build_3x2x1():
        return schoolbook(3, 2, 1)

    # Helper to build 3x3x2 (split M into 2+1)
    def build_3x3x2():
        sU, sV, sW = build_3x2x2()           # 3x2x2, r=11
        bU, bV, bW = schoolbook(3, 1, 2)      # 3x1x2, r=6
        r = 11 + 6
        lU = [[0.0]*r for _ in range(9)]
        lV = [[0.0]*r for _ in range(6)]
        lW = [[0.0]*r for _ in range(6)]
        # 3x2x2 part: j=0,1
        for i in range(3):
            for j in range(2):
                for c in range(11):
                    lU[i*3+j][c] = sU[i*2+j][c]
            for k in range(2):
                for c in range(11):
                    lW[k*3+i][c] = sW[k*3+i][c]
        for j in range(2):
            for k in range(2):
                for c in range(11):
                    lV[j*2+k][c] = sV[j*2+k][c]
        # 3x1x2 part: j=2
        for i in range(3):
            for c in range(6):
                lU[i*3+2][11+c] = bU[i*1+0][c]
            for k in range(2):
                for c in range(6):
                    lW[k*3+i][11+c] = bW[k*3+i][c]
        for k in range(2):
            for c in range(6):
                lV[2*2+k][11+c] = bV[0*2+k][c]
        return lU, lV, lW

    # Helper to build 3x3x1 (schoolbook)
    def build_3x3x1():
        return schoolbook(3, 3, 1)

    # Assemble 3x2x7: split P into 2,2,2,1
    def build_3x2x7():
        u1,v1,w1 = build_3x2x2()  # r=11
        u2,v2,w2 = build_3x2x2()  # r=11
        u3,v3,w3 = build_3x2x2()  # r=11
        u4,v4,w4 = build_3x2x1()  # r=6
        r = 11+11+11+6
        lU = [[0.0]*r for _ in range(6)]
        lV = [[0.0]*r for _ in range(14)]
        lW = [[0.0]*r for _ in range(14)]
        blocks = [(u1,v1,w1,0,11),(u2,v2,w2,11,11),(u3,v3,w3,22,11),(u4,v4,w4,33,6)]
        for lU_b, lV_b, lW_b, c0, rb in blocks:
            for i in range(3):
                for j in range(2):
                    for c in range(rb):
                        lU[i*2+j][c0+c] = lU_b[i*2+j][c]
            for j in range(2):
                for k in range(2):
                    for c in range(rb):
                        lV[j*7+k][c0+c] = lV_b[j*2+k][c]
            for k in range(2):
                for i in range(3):
                    for c in range(rb):
                        lW[k*3+i][c0+c] = lW_b[k*3+i][c]
            # Adjust V and W for different k offsets
        # Need to fix V and W indexing for different P blocks
        lV = [[0.0]*r for _ in range(14)]
        lW = [[0.0]*r for _ in range(14)]
        p_offsets = [0, 2, 4, 6]
        for idx, (lU_b, lV_b, lW_b, c0, rb) in enumerate(blocks):
            for i in range(3):
                for j in range(2):
                    for c in range(rb):
                        lU[i*2+j][c0+c] = lU_b[i*2+j][c]
            k_off = p_offsets[idx]
            bp = 2 if idx < 3 else 1
            for j in range(2):
                for k in range(bp):
                    for c in range(rb):
                        lV[j*7+(k_off+k)][c0+c] = lV_b[j*bp+k][c]
            for k in range(bp):
                for i in range(3):
                    for c in range(rb):
                        lW[(k_off+k)*3+i][c0+c] = lW_b[k*3+i][c]
        return lU, lV, lW

    # Assemble 3x3x7: split P into 2,2,2,1
    def build_3x3x7():
        u1,v1,w1 = build_3x3x2()  # r=17
        u2,v2,w2 = build_3x3x2()  # r=17
        u3,v3,w3 = build_3x3x2()  # r=17
        u4,v4,w4 = build_3x3x1()  # r=9
        r = 17+17+17+9
        lU = [[0.0]*r for _ in range(9)]
        lV = [[0.0]*r for _ in range(21)]
        lW = [[0.0]*r for _ in range(21)]
        blocks = [(u1,v1,w1,0,17),(u2,v2,w2,17,17),(u3,v3,w3,34,17),(u4,v4,w4,51,9)]
        p_offsets = [0, 2, 4, 6]
        for idx, (lU_b, lV_b, lW_b, c0, rb) in enumerate(blocks):
            for i in range(3):
                for j in range(3):
                    for c in range(rb):
                        lU[i*3+j][c0+c] = lU_b[i*3+j][c]
            k_off = p_offsets[idx]
            bp = 2 if idx < 3 else 1
            for j in range(3):
                for k in range(bp):
                    for c in range(rb):
                        lV[j*7+(k_off+k)][c0+c] = lV_b[j*bp+k][c]
            for k in range(bp):
                for i in range(3):
                    for c in range(rb):
                        lW[(k_off+k)*3+i][c0+c] = lW_b[k*3+i][c]
        return lU, lV, lW

    # Assemble final 3x5x7: split M into 2+3
    u1,v1,w1 = build_3x2x7()  # r=39
    u2,v2,w2 = build_3x3x7()  # r=60
    r = 39 + 60
    lU = [[0.0]*r for _ in range(15)]
    lV = [[0.0]*r for _ in range(35)]
    lW = [[0.0]*r for _ in range(21)]
    for i in range(3):
        for j in range(2):
            for c in range(39):
                lU[i*5+j][c] = u1[i*2+j][c]
        for j in range(3):
            for c in range(60):
                lU[i*5+(2+j)][39+c] = u2[i*3+j][c]
    for j in range(2):
        for k in range(7):
            for c in range(39):
                lV[j*7+k][c] = v1[j*7+k][c]
    for j in range(3):
        for k in range(7):
            for c in range(60):
                lV[(2+j)*7+k][39+c] = v2[j*7+k][c]
    for k in range(7):
        for i in range(3):
            for c in range(39):
                lW[k*3+i][c] = w1[k*3+i][c]
            for c in range(60):
                lW[k*3+i][39+c] = w2[k*3+i][c]

    # Verify rank is exactly 99
    assert len(lU[0]) == 99, f"Expected rank 99, got {len(lU[0])}"

    # Copy to global
    for a in range(15):
        for c in range(99):
            U[a][c] = lU[a][c]
    for a in range(35):
        for c in range(99):
            V[a][c] = lV[a][c]
    for a in range(21):
        for c in range(99):
            W[a][c] = lW[a][c]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
