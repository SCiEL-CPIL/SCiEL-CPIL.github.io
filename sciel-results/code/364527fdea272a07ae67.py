"""Matrix multiplication 2,2,10 - rank-35 Strassen-based decomposition."""
import time

DIMENSIONS = [2, 2, 10]
_DEADLINE = time.time() + 140


def construct():
    """Construct a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    rank = 35

    U = [[0] * rank for _ in range(n * m)]   # (4, 35)
    V = [[0] * rank for _ in range(m * p)]   # (20, 35)
    W = [[0] * rank for _ in range(p * n)]   # (20, 35)

    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # A indices: 0=a(A[0,0]), 1=b(A[0,1]), 2=c(A[1,0]), 3=d(A[1,1])
    # B indices: 0=e(B[0,0]), 1=f(B[0,1]), 2=g(B[1,0]), 3=h(B[1,1])
    #
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # Result:
    # C[0,0] = P1 + P4 - P5 + P7
    # C[0,1] = P3 + P5
    # C[1,0] = P2 + P4
    # C[1,1] = P1 - P2 + P3 + P6

    # U coefficients: [a_coeff, b_coeff, c_coeff, d_coeff] for each product
    strassen_A = [
        [1, 0, 0, 1],   # P1: a+d
        [0, 0, 1, 1],   # P2: c+d
        [1, 0, 0, 0],   # P3: a
        [0, 0, 0, 1],   # P4: d
        [1, 1, 0, 0],   # P5: a+b
        [-1, 0, 1, 0],  # P6: c-a
        [0, 1, 0, -1],  # P7: b-d
    ]

    # V coefficients: [e_coeff, f_coeff, g_coeff, h_coeff] for each product
    strassen_B = [
        [1, 0, 0, 1],   # P1: e+h
        [1, 0, 0, 0],   # P2: e
        [0, 1, 0, -1],  # P3: f-h
        [-1, 0, 1, 0],  # P4: g-e
        [0, 0, 0, 1],   # P5: h
        [1, 1, 0, 0],   # P6: e+f
        [0, 0, 1, 1],   # P7: g+h
    ]

    # W coefficients: [C00_coeff, C01_coeff, C10_coeff, C11_coeff] for each product
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    w_contrib = [
        [1, 0, 0, 0],   # P1: C00+1, C11+1
        [0, 0, 1, -1],  # P2: C10+1, C11-1
        [0, 1, 0, 1],   # P3: C01+1, C11+1
        [1, 0, 1, 0],   # P4: C00+1, C10+1
        [-1, 1, 0, 0],  # P5: C00-1, C01+1
        [0, 0, 0, 1],   # P6: C11+1
        [1, 0, 0, 0],   # P7: C00+1
    ]
    # Fix: P1 also contributes to C11
    w_contrib[0][3] = 1

    # For each of the 5 column-blocks b (columns 2b and 2b+1 of B and C):
    for b in range(5):
        for r in range(7):
            col = b * 7 + r

            # U: index i*m+j
            # a=A[0,0]->idx0, b=A[0,1]->idx1, c=A[1,0]->idx2, d=A[1,1]->idx3
            U[0][col] = strassen_A[r][0]
            U[1][col] = strassen_A[r][1]
            U[2][col] = strassen_A[r][2]
            U[3][col] = strassen_A[r][3]

            # V: index j*p+k
            # e=B[0,2b]->idx 0*p+2b=2b
            # f=B[0,2b+1]->idx 0*p+2b+1=2b+1
            # g=B[1,2b]->idx 1*p+2b=10+2b
            # h=B[1,2b+1]->idx 1*p+2b+1=10+2b+1
            k0 = 2 * b
            k1 = 2 * b + 1
            V[0 * p + k0][col] = strassen_B[r][0]
            V[0 * p + k1][col] = strassen_B[r][1]
            V[1 * p + k0][col] = strassen_B[r][2]
            V[1 * p + k1][col] = strassen_B[r][3]

            # W: index k*n+i (official trace convention)
            # C00: k=2b, i=0 -> idx (2b)*2+0
            # C01: k=2b+1, i=0 -> idx (2b+1)*2+0
            # C10: k=2b, i=1 -> idx (2b)*2+1
            # C11: k=2b+1, i=1 -> idx (2b+1)*2+1
            W[(2 * b) * n + 0][col] = w_contrib[r][0]
            W[(2 * b + 1) * n + 0][col] = w_contrib[r][1]
            W[(2 * b) * n + 1][col] = w_contrib[r][2]
            W[(2 * b + 1) * n + 1][col] = w_contrib[r][3]

    return {"U": U, "V": V, "W": W}
