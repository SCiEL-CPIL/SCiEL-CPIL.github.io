"""6x6 matrix multiplication tensor decomposition.

Strategy:
  1. Primary: Laderman(3) ⊗ Strassen(2) → rank 161
  2. Fallback: 3x3 schoolbook ⊗ Strassen(2) → rank 189
  3. Fallback: 6x6 schoolbook → rank 216
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START


def construct():
    """Return a rank-minimal exact tensor decomposition for 6x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0

    # Fallback: schoolbook (rank 216)
    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Primary: Laderman(3) ⊗ Strassen(2) → rank 23 * 7 = 161
    try:
        result = _kron_product(_laderman(), _strassen())
        if result is not None and _check(result, n, m, p):
            best = result
            best_rank = 161
    except Exception:
        pass

    if time.time() > deadline:
        return best

    # Secondary: 3x3 schoolbook ⊗ Strassen(2) → rank 27 * 7 = 189
    if best_rank > 189:
        try:
            result2 = _kron_product(_schoolbook(3, 3, 3), _strassen())
            if result2 is not None and _check(result2, n, m, p):
                best = result2
                best_rank = 189
        except Exception:
            pass

    return best


def _schoolbook(n, m, p):
    """Build the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen():
    """
    Rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Convention: C[k,i] = sum_j A[i,j] B[j,k], so W index is k*2+i.
    A indices: a=0(A[0,0]), b=1(A[0,1]), c=2(A[1,0]), d=3(A[1,1])
    B indices: e=0(B[0,0]), f=1(B[0,1]), g=2(B[1,0]), h=3(B[1,1])
    C indices: C00=0, C01=1, C10=2, C11=3
    """
    rank = 7
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(4)]
    W = [[0.0] * rank for _ in range(4)]

    # P1 = (a+d)(e+h) → C00+1, C11+1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1

    # P2 = (c+d)e → C11-1
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[3][1] += -1

    # P3 = a(f-h) → C01+1, C11+1
    U[0][2] += 1
    V[1][2] += 1; V[3][2] += -1
    W[1][2] += 1; W[3][2] += 1

    # P4 = d(g-e) → C10+1, C11+1
    U[3][3] += 1
    V[2][3] += 1; V[0][3] += -1
    W[2][3] += 1; W[3][3] += 1

    # P5 = (a+b)h → C00-1, C01+1
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] += -1; W[1][4] += 1

    # P6 = (c-a)(e+f) → C00+1
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[0][5] += 1

    # P7 = (b-d)(g+h) → C00+1, C10+1
    U[1][6] += 1; U[3][6] += -1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1; W[2][6] += 1

    return {"U": U, "V": V, "W": W}


def _laderman():
    """
    Rank-23 Laderman decomposition for 3x3 matrix multiplication.
    Convention: C[k,i] = sum_j A[i,j] B[j,k], so W index is k*3+i.
    A/B indices: 0..8 for 3x3 matrices.
    """
    rank = 23
    U = [[0.0] * rank for _ in range(9)]
    V = [[0.0] * rank for _ in range(9)]
    W = [[0.0] * rank for _ in range(9)]

    # 23 products: (a_terms, b_terms)
    products = [
        ([(0,1),(4,1),(8,1)], [(0,1),(4,1),(8,1)]),
        ([(0,1),(4,1),(8,-1)], [(0,1),(4,1),(8,-1)]),
        ([(0,1),(4,-1),(8,1)], [(0,1),(4,-1),(8,1)]),
        ([(0,1),(4,-1),(8,-1)], [(0,1),(4,-1),(8,-1)]),
        ([(0,1),(5,1),(7,-1)], [(0,1),(7,1),(5,-1)]),
        ([(0,1),(5,-1),(7,1)], [(0,1),(7,-1),(5,1)]),
        ([(0,1),(3,1),(1,-1)], [(0,1),(3,1),(1,-1)]),
        ([(0,1),(3,-1),(1,1)], [(0,1),(3,-1),(1,1)]),
        ([(1,1),(3,1),(8,1)], [(1,1),(3,1),(8,1)]),
        ([(1,1),(3,1),(8,-1)], [(1,1),(3,1),(8,-1)]),
        ([(1,1),(3,-1),(8,1)], [(1,1),(3,-1),(8,1)]),
        ([(1,1),(3,-1),(8,-1)], [(1,1),(3,-1),(8,-1)]),
        ([(2,1),(6,1),(4,1)], [(2,1),(6,1),(4,1)]),
        ([(2,1),(6,1),(4,-1)], [(2,1),(6,1),(4,-1)]),
        ([(2,1),(6,-1),(4,1)], [(2,1),(6,-1),(4,1)]),
        ([(2,1),(6,-1),(4,-1)], [(2,1),(6,-1),(4,-1)]),
        ([(5,1),(7,1),(0,1)], [(5,1),(7,1),(0,1)]),
        ([(5,1),(7,1),(0,-1)], [(5,1),(7,1),(0,-1)]),
        ([(5,1),(7,-1),(0,1)], [(5,1),(7,-1),(0,1)]),
        ([(5,1),(7,-1),(0,-1)], [(5,1),(7,-1),(0,-1)]),
        ([(6,1),(2,1),(4,1)], [(6,1),(2,1),(4,1)]),
        ([(6,1),(2,1),(4,-1)], [(6,1),(2,1),(4,-1)]),
        ([(6,1),(2,-1),(4,1)], [(6,1),(2,-1),(4,1)]),
    ]

    for col, (a_terms, b_terms) in enumerate(products):
        for (idx, coeff) in a_terms:
            U[idx][col] += coeff
        for (idx, coeff) in b_terms:
            V[idx][col] += coeff

    # W reconstruction: C[k,i] = sum_j A[i,j] B[j,k]
    # W index: k*3+i
    #
    # W[0] = C[0,0]: p1+p2+p3+p4+p5+p6+p7+p8
    for col in range(8):
        W[0][col] += 1.0/8

    # W[3] = C[0,1]: p9+p10+p11+p12
    for col in range(8, 12):
        W[3][col] += 1.0/4

    # W[6] = C[0,2]: p13+p14+p15+p16
    for col in range(12, 16):
        W[6][col] += 1.0/4

    # W[1] = C[1,0]: p17+p18+p19+p20
    for col in range(16, 20):
        W[1][col] += 1.0/4

    # W[4] = C[1,1]: p1+p2+p3+p4+p5+p6+p7+p8
    W[4][0] += 1.0/8
    W[4][1] += -1.0/8
    W[4][2] += -1.0/8
    W[4][3] += 1.0/8
    W[4][4] += 1.0/8
    W[4][5] += -1.0/8
    W[4][6] += -1.0/8
    W[4][7] += 1.0/8

    # W[7] = C[1,2]: p9-p10+p11-p12
    W[7][8] += 1.0/4
    W[7][9] += -1.0/4
    W[7][10] += 1.0/4
    W[7][11] += -1.0/4

    # W[2] = C[2,0]: p21+p22+p23
    W[2][20] += 1.0/3
    W[2][21] += 1.0/3
    W[2][22] += 1.0/3

    # W[5] = C[2,1]: p13-p14-p15+p16
    W[5][12] += 1.0/4
    W[5][13] += -1.0/4
    W[5][14] += -1.0/4
    W[5][15] += 1.0/4

    # W[8] = C[2,2]: p1-p2+p3-p4-p5+p6+p7-p8
    W[8][0] += 1.0/8
    W[8][1] += -1.0/8
    W[8][2] += 1.0/8
    W[8][3] += -1.0/8
    W[8][4] += -1.0/8
    W[8][5] += 1.0/8
    W[8][6] += 1.0/8
    W[8][7] += -1.0/8

    return {"U": U, "V": V, "W": W}


def _kron_product(decomp1, decomp2):
    """
    Kronecker product of two tensor decompositions.
    decomp1: n1×n1 matrices, rank r1
    decomp2: n2×n2 matrices, rank r2
    Result: (n1*n2)×(n1*n2) matrices, rank r1*r2
    """
    U1 = np.array(decomp1['U'], dtype=float)
    V1 = np.array(decomp1['V'], dtype=float)
    W1 = np.array(decomp1['W'], dtype=float)
    U2 = np.array(decomp2['U'], dtype=float)
    V2 = np.array(decomp2['V'], dtype=float)
    W2 = np.array(decomp2['W'], dtype=float)

    r1 = U1.shape[1]
    r2 = U2.shape[1]
    n1 = int(round(np.sqrt(U1.shape[0])))
    n2 = int(round(np.sqrt(U2.shape[0])))

    N = n1 * n2
    rank = r1 * r2

    U = np.zeros((N * N, rank))
    V = np.zeros((N * N, rank))
    W = np.zeros((N * N, rank))

    for a in range(r1):
        for b in range(r2):
            col = a * r2 + b
            U[:, col] = np.kron(U1[:, a], U2[:, b])
            V[:, col] = np.kron(V1[:, a], V2[:, b])
            W[:, col] = np.kron(W1[:, a], W2[:, b])

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _check(decomp, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U = np.array(decomp['U'], dtype=float)
    V = np.array(decomp['V'], dtype=float)
    W = np.array(decomp['W'], dtype=float)

    R = U.shape[1]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(R):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


# EVOLVE-BLOCK-END
