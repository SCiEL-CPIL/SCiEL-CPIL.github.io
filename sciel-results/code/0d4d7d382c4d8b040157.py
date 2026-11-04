"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS  # 4, 5, 6

    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Schoolbook fallback
    def schoolbook():
        n, m, p = DIMENSIONS
        R = n*m*p
        U = [[0.0]*R for _ in range(n*m)]
        V = [[0.0]*R for _ in range(m*p)]
        W = [[0.0]*R for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1.0
                    V[j*p+k][c] = 1.0
                    W[k*n+i][c] = 1.0
                    c += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best = (U_sb, V_sb, W_sb, 120)

    if time.time() > deadline - 5:
        return {"U": best[0], "V": best[1], "W": best[2]}

    # 2x2x2 Strassen (rank 7)
    def build_strassen_2x2():
        U = [[0.0]*7 for _ in range(4)]
        V = [[0.0]*7 for _ in range(4)]
        W = [[0.0]*7 for _ in range(4)]
        # terms: (A coeffs, B coeffs, C coeffs) with (row,col) 0-indexed
        terms = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),
            ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1)]),
            ([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,-1),(1,0,1)]),
            ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,1),(0,1,1)]),
            ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
            ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,-1),(1,1,1)]),
        ]
        for s, (ac, bc, cc) in enumerate(terms):
            for (r,c,v) in ac: U[r*2+c][s] += v
            for (r,c,v) in bc: V[r*2+c][s] += v
            for (r,c,v) in cc: W[r*2+c][s] += v
        return U, V, W

    U2, V2, W2 = build_strassen_2x2()

    # verify 2x2
    def verify2(U,V,W):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U[i*2+j][r]*V[j*2+k][r]*W[k*2+i][r] for r in range(7))
                    if abs(s-1.0) > 1e-9:
                        return False
        return True

    if not verify2(U2, V2, W2):
        # fallback: don't use Strassen
        return {"U": best[0], "V": best[1], "W": best[2]}

    # Tensor square -> 4x4x4 rank 49
    def tensor_square(U2, V2, W2):
        R = 49
        U4 = [[0.0]*R for _ in range(16)]
        V4 = [[0.0]*R for _ in range(16)]
        W4 = [[0.0]*R for _ in range(16)]
        for i1 in range(2):
            for i2 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        for s1 in range(7):
                            for s2 in range(7):
                                col = s1*7+s2
                                u = U2[i1*2+j1][s1]*U2[i2*2+j2][s2]
                                U4[(i1*2+i2)*4 + (j1*2+j2)][col] += u
        for j1 in range(2):
            for j2 in range(2):
                for k1 in range(2):
                    for k2 in range(2):
                        for s1 in range(7):
                            for s2 in range(7):
                                col = s1*7+s2
                                v = V2[j1*2+k1][s1]*V2[j2*2+k2][s2]
                                V4[(j1*2+j2)*4 + (k1*2+k2)][col] += v
        for k1 in range(2):
            for k2 in range(2):
                for i1 in range(2):
                    for i2 in range(2):
                        for s1 in range(7):
                            for s2 in range(7):
                                col = s1*7+s2
                                w = W2[k1*2+i1][s1]*W2[k2*2+i2][s2]
                                W4[(k1*2+k2)*4 + (i1*2+i2)][col] += w
        return U4, V4, W4

    U4, V4, W4 = tensor_square(U2, V2, W2)

    def verify4(U,V,W):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = sum(U[i*4+j][r]*V[j*4+k][r]*W[k*4+i][r] for r in range(49))
                    if abs(s-1.0) > 1e-9:
                        return False
        return True

    if not verify4(U4, V4, W4):
        return {"U": best[0], "V": best[1], "W": best[2]}

    # Assemble 4x5x6, rank 49+32+24 = 105
    R = 105
    U = [[0.0]*R for _ in range(20)]
    V = [[0.0]*R for _ in range(30)]
    W = [[0.0]*R for _ in range(24)]

    # Part 1: A1(4x4) x B1a(4x4) -> C[0:4,0:4], cols 0..48
    for col in range(49):
        for i in range(4):
            for j in range(4):
                if U4[i*4+j][col] != 0:
                    U[i*5+j][col] += U4[i*4+j][col]
        for j in range(4):
            for k in range(4):
                if V4[j*4+k][col] != 0:
                    V[j*6+k][col] += V4[j*4+k][col]
        for k in range(4):
            for i in range(4):
                if W4[k*4+i][col] != 0:
                    W[k*4+i][col] += W4[k*4+i][col]

    # Part 2: A1(4x4) x B1b(4x2) -> C[4:6,0:4], cols 49..80
    off = 49
    for i in range(4):
        for j in range(4):
            for kk in range(2):
                col = off + i*8 + j*2 + kk
                U[i*5+j][col] = 1.0
                V[j*6 + (4+kk)][col] = 1.0
                W[(4+kk)*4 + i][col] = 1.0

    # Part 3: A2(4x1) x B2(1x6) -> C[0:6,0:4], cols 81..104
    off = 81
    for i in range(4):
        for k in range(6):
            col = off + i*6 + k
            U[i*5+4][col] = 1.0
            V[4*6+k][col] = 1.0
            W[k*4+i][col] = 1.0

    if verify(U, V, W):
        best = (U, V, W, 105)

    return {"U": best[0], "V": best[1], "W": best[2]}
# EVOLVE-BLOCK-END
