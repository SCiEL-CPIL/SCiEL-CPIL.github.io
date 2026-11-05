"""Matrix multiplication 2x4x5 via row-split + Strassen/2x2x3 decomposition.

Split the 4 input columns into two halves (j in {0,1} and j in {2,3}).  Each
half is a 2x2x5 problem split into a 2x2x2 (Strassen, rank 7), a 2x2x3
(rank-8 identity) and a 2x2x1 (schoolbook, rank 4).  Total rank = 2*(7+8+4)=38
< 40 schoolbook.
"""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 2, 4, 5

    S = [
        (1, 1, 1), (-1, -1, 1), (1, 0, -1), (0, 1, 1),
        (1, 1, -1), (-1, 0, 1), (0, -1, -1),
    ]
    def S2():
        U = [[0] * 7 for _ in range(2)]
        V = [[0] * 7 for _ in range(2)]
        W = [[0] * 7 for _ in range(4)]  # k*n+i, k in {0,1}
        for c, (u0, u1, w) in enumerate(S):
            U[0][c] = u0
            U[1][c] = u1
            V[0][c] = 1
            V[1][c] = 1
            W[0][c] = w
            W[1][c] = w
            W[2][c] = 0
            W[3][c] = 0
        return U, V, W

    def S23():
        U = [[0] * 8 for _ in range(2)]
        V = [[0] * 8 for _ in range(2)]
        W = [[0] * 8 for _ in range(6)]  # k*n+i, k in {0,1,2}
        for c, (u0, u1, w) in enumerate(S):
            U[0][c] = u0
            U[1][c] = u1
            V[0][c] = 1
            V[1][c] = 1
            W[0][c] = w
            W[1][c] = w
        for a in range(2):
            for b in range(2):
                c = 4 + a * 2 + b
                U[a][c] = 1
                V[b][c] = 1
                W[2 * a + b][c] = 1
        return U, V, W

    def s221():
        U = [[0] * 4 for _ in range(2)]
        V = [[0] * 4 for _ in range(2)]
        W = [[0] * 4 for _ in range(2)]  # k*n+i, k in {0}
        for i in range(2):
            for j in range(2):
                c = i * 2 + j
                U[i][c] = 1
                V[j][c] = 1
                W[i][c] = 1
        return U, V, W

    RANK = 38
    U = [[0] * RANK for _ in range(8)]   # n*m = 8 rows
    V = [[0] * RANK for _ in range(20)]  # m*p = 20 rows
    W = [[0] * RANK for _ in range(10)]  # p*n = 10 rows

    def place(Uf, Vf, Wf, uoff, voff, woff, base):
        for i in range(len(Uf)):
            for c in range(len(Uf[i])):
                U[uoff + i][base + c] = Uf[i][c]
        for j in range(len(Vf)):
            for c in range(len(Vf[j])):
                V[voff + j][base + c] = Vf[j][c]
        for k in range(len(Wf)):
            for c in range(len(Wf[k])):
                W[woff + k][base + c] = Wf[k][c]

    # Upper half: i in {0,1}
    Ua, Va, Wa = S2()    # j in {0,1}, k in {0,1}
    place(Ua, Va, Wa, uoff=0, voff=0,  woff=0, base=0)
    Ub, Vb, Wb = S23()   # j in {0,1}, k in {2,3}
    place(Ub, Vb, Wb, uoff=0, voff=0,  woff=4, base=7)
    Uc, Vc, Wc = s221()  # j in {0,1}, k in {4}
    place(Uc, Vc, Wc, uoff=0, voff=0,  woff=8, base=15)

    # Lower half: i in {2,3}
    Ud, Vd, Wd = S2()    # j in {2,3}, k in {0,1}
    place(Ud, Vd, Wd, uoff=4, voff=8,  woff=2, base=19)
    Ue, Ve, We = S23()   # j in {2,3}, k in {2,3}
    place(Ue, Ve, We, uoff=4, voff=8,  woff=6, base=26)
    Uf, Vf, Wf = s221()  # j in {2,3}, k in {4}
    place(Uf, Vf, Wf, uoff=4, voff=8,  woff=10, base=34)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
