"""Matrix multiplication 2x4 x 4x4: rank-minimal exact tensor decomposition.

Uses a Strassen-based block decomposition: split the 4x4 B into 2x2 blocks and
the 2x4 A into two 2x2 blocks. The product C = A*B is then computed as two
2x2 block-sums, each handled by two Strassen 2x2x2x2 multiplications (7 each),
giving total rank 28 (vs. schoolbook 32).
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120 - 15  # safety margin before the 120s kill
    n, m, p = DIMENSIONS

    terms = []  # list of (alpha: {(i,j):v}, beta: {(j,k):v}, w: {(i,k):v})

    def add_term(alpha, beta, w):
        terms.append((alpha, beta, w))

    def strassen(xmap, ymap, wmap):
        # M0 = (X00+X11)(Y00+Y11) -> C[0,0]+=1, C[1,1]+=1
        add_term({xmap(0,0):1, xmap(1,1):1},
                 {ymap(0,0):1, ymap(1,1):1},
                 {wmap(0,0):1, wmap(1,1):1})
        # M1 = (X10+X11)Y00 -> C[0,0]-=1
        add_term({xmap(1,0):1, xmap(1,1):1},
                 {ymap(0,0):1},
                 {wmap(0,0):-1})
        # M2 = X00(Y10-Y11) -> C[0,1]+=1
        add_term({xmap(0,0):1},
                 {ymap(1,0):1, ymap(1,1):-1},
                 {wmap(0,1):1})
        # M3 = X11(Y01-Y00) -> C[1,0]+=1
        add_term({xmap(1,1):1},
                 {ymap(0,1):1, ymap(0,0):-1},
                 {wmap(1,0):1})
        # M4 = (X00+X01)Y11 -> C[1,1]-=1
        add_term({xmap(0,0):1, xmap(0,1):1},
                 {ymap(1,1):1},
                 {wmap(1,1):-1})
        # M5 = (X10-X00)(Y00+Y01) -> C[1,1]+=1
        add_term({xmap(1,0):1, xmap(0,0):-1},
                 {ymap(0,0):1, ymap(0,1):1},
                 {wmap(1,1):1})
        # M6 = (X01-X11)(Y10+Y11) -> C[0,1]+=1
        add_term({xmap(0,1):1, xmap(1,1):-1},
                 {ymap(1,0):1, ymap(1,1):1},
                 {wmap(0,1):1})

    # Left half of C (cols 0,1): C_left = A1*B11 + A2*B21
    #   A1[r,c]=A[r,c]; A2[r,c]=A[r,c+2]; B11[r,c]=B[r,c]; B21[r,c]=B[r+2,c]
    strassen(lambda r,c:(r,c),     lambda r,c:(r,c),     lambda r,c:(r,c))
    strassen(lambda r,c:(r,c+2),   lambda r,c:(r+2,c),   lambda r,c:(r,c))
    # Right half of C (cols 2,3): C_right = A1*B12 + A2*B22
    #   B12[r,c]=B[r,c+2]; B22[r,c]=B[r+2,c+2]
    strassen(lambda r,c:(r,c),     lambda r,c:(r,c+2),   lambda r,c:(r,c+2))
    strassen(lambda r,c:(r,c+2),   lambda r,c:(r+2,c+2), lambda r,c:(r,c+2))

    rank = len(terms)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for c, (alpha, beta, w) in enumerate(terms):
        for (i,j), v in alpha.items():
            U[i*m+j][c] = v
        for (j,k), v in beta.items():
            V[j*p+k][c] = v
        for (i,k), v in w.items():
            W[k*n+i][c] = v

    # Fast vectorized verification with random integer matrices.
    if time.time() < deadline:
        U3 = np.array(U).reshape(n, m, rank)
        V3 = np.array(V).reshape(m, p, rank)
        W3 = np.array(W).reshape(p, n, rank)
        rng = np.random.default_rng(12345)
        for _ in range(200):
            A = rng.integers(-9, 9, size=(n, m))
            B = rng.integers(-9, 9, size=(m, p))
            LA = np.einsum('ijc,ij->ci', U3, A)   # (rank, n)
            LB = np.einsum('jkc,jk->ck', V3, B)   # (rank, p)
            C = np.einsum('ci,ck,kic->ik', LA, LB, W3)
            if not np.array_equal(C, A @ B):
                raise ValueError("decomposition verification failed")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
