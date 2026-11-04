"""Rewritten program: Kronecker product of Strassen rank-7 decomposition.

Yields rank 49 for 4x4 * 4x4 matrix multiplication (vs 64 schoolbook),
using the Kronecker product of Strassen's rank-7 2x2 decomposition.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time


def _strassen_2x2():
    # Strassen rank-7 decomposition for 2x2 * 2x2.
    # U[(i,j),c] at row i*2+j ; V[(j,k),c] at row j*2+k ; W[(k,i),c] at row k*2+i.
    # Identity: sum_c U[i,j,c] * V[j,k,c] == W[i,k,c].
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]

    # c=0: (a+b)(e+f) -> C11, C21, C22
    U[0][0] = 1; U[1][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[2][0] = 1; W[3][0] = 1

    # c=1: (c+d)(e+f) -> C12, C22
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1; V[3][1] = 1
    W[1][1] = 1; W[3][1] = 1

    # c=2: a(d-f) -> C21
    U[0][2] = 1
    V[3][2] = 1; V[2][2] = -1
    W[2][2] = 1

    # c=3: c(a-b) -> C11
    U[2][3] = 1; U[0][3] = 1; U[1][3] = -1
    W[0][3] = 1

    # c=4: (a+c)(e+d) -> C11, C12
    U[0][4] = 1; U[2][4] = 1
    V[0][4] = 1; V[2][4] = 1
    W[0][4] = 1; W[1][4] = 1

    # c=5: (b+d)(f-e) -> C12, C22
    U[1][5] = 1; U[3][5] = 1
    V[3][5] = 1; V[0][5] = -1
    W[1][5] = 1; W[3][5] = 1

    # c=6: (a-c)(e+b) -> C21
    U[0][6] = 1; U[2][6] = -1
    V[0][6] = 1; V[1][6] = 1
    W[2][6] = 1

    return U, V, W


def _kron(U, V):
    # U: n1 x r1, V: n2 x r2 -> (n1*n2) x (r1*r2)
    # out[(i1*n2+i2)][c1*r2+c2] = U[i1][c1] * V[i2][c2]
    n1 = len(U); r1 = len(U[0]); n2 = len(V); r2 = len(V[0])
    out = [[0] * (r1 * r2) for _ in range(n1 * n2)]
    for i1 in range(n1):
        for c1 in range(r1):
            a = U[i1][c1]
            if a == 0:
                continue
            base = i1 * n2
            for i2 in range(n2):
                for c2 in range(r2):
                    b = V[i2][c2]
                    if b == 0:
                        continue
                    out[base + i2][c1 * r2 + c2] = a * b
    return out


def construct():
    n, m, p = DIMENSIONS
    # 4 = 2 x 2, so Kronecker product of the rank-7 2x2 decomposition.
    sU, sV, sW = _strassen_2x2()
    U = _kron(sU, sU)
    V = _kron(sV, sV)
    W = _kron(sW, sW)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
